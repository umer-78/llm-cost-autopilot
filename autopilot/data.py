"""The measured answers the router learns from: HELM Lite's public per-question results.

HELM (Stanford CRFM) ran the same prompts through each model and published every answer,
whether it was right, and how many tokens it took. Six tasks here, picked because each
answer is right or wrong: grade-school maths (GSM8K), competition maths (MATH, level 1),
MMLU, MedQA, OpenBookQA and LegalBench. Downloaded on first use into AUTOPILOT_DATA
(default ~/.cache/autopilot) and reduced to one table: for each question, each model's
answer, whether it was right, and its token counts.
"""
import hashlib
import json
import os
import re
import tarfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import yaml

BUCKET = "https://storage.googleapis.com/crfm-helm-public"
LISTING = "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o"
REGISTRY = Path(__file__).with_name("models.yaml")
MODELS = {   # the sentence embedder for the router's features: (url, sha256, folder, onnx file, pooling)
    "bge-small": ("https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz",
                  "3858004b3822f64f940280874b8f2d2dc25b34a4f3eb3cdf617bdceeb21ed9ed", "fast-bge-small-en-v1.5", "model_optimized.onnx", "cls"),
}
TASKS = {   # HELM scenario -> (task name, the metric that says right or wrong)
    "gsm": ("gsm8k", "final_number_exact_match"),
    "math": ("math", "math_equiv_chain_of_thought"),
    "mmlu": ("mmlu", "exact_match"),
    "med_qa": ("medqa", "exact_match"),
    "commonsense": ("openbookqa", "exact_match"),
    "legalbench": ("legalbench", "quasi_exact_match"),
}


def registry():
    return yaml.safe_load(REGISTRY.read_text())


def cache_dir() -> Path:
    path = Path(os.environ.get("AUTOPILOT_DATA", Path.home() / ".cache" / "autopilot"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_json(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return json.loads(path.read_text())


def runs_for(helm_model, release):
    """The HELM run names for one model's six tasks."""
    names, token = [], None
    while True:
        q = {"prefix": f"lite/benchmark_output/runs/{release}/", "delimiter": "/", "maxResults": 1000}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(f"{LISTING}?{urllib.parse.urlencode(q)}", timeout=120) as r:
            page = json.load(r)
        names += [p.rstrip("/").rsplit("/", 1)[1] for p in page.get("prefixes", [])]
        token = page.get("nextPageToken")
        if not token:
            break
    mine = [n for n in names if re.search(rf"(?:^|[,:])model={re.escape(helm_model)}(?:,|$)", n)]
    return [n for n in mine if n.split(":")[0] in TASKS]


def download(release, run):
    folder = cache_dir() / "helm" / release / urllib.parse.quote(run, safe="")
    folder.mkdir(parents=True, exist_ok=True)
    base = f"{BUCKET}/lite/benchmark_output/runs/{release}/{urllib.parse.quote(run, safe='')}"
    return (fetch_json(f"{base}/display_predictions.json", folder / "display_predictions.json"),
            fetch_json(f"{base}/instances.json", folder / "instances.json"))


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalise(scenario, prediction):
    """The part of an answer two models can agree on: a letter, a number, a label."""
    text = (prediction.get("predicted_text") or "").strip()
    if scenario in ("mmlu", "med_qa", "commonsense"):
        return (text[:1] or "").upper()
    if scenario == "gsm":
        tail = text.split("The answer is")[-1]
        numbers = NUMBER.findall(tail) or NUMBER.findall(text)
        return numbers[-1].replace(",", "").rstrip(".") if numbers else ""
    if scenario == "math":
        boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        answer = boxed[-1] if boxed else text.split("The answer is")[-1]
        return re.sub(r"[\s$.]", "", answer).lower()
    return text.splitlines()[0].strip().lower() if text else ""


def subtask_of(run):
    args = dict(a.split("=", 1) for a in run.split(":", 1)[1].split(",") if "=" in a)
    return args.get("subject") or args.get("subset") or args.get("dataset") or ""


@lru_cache(maxsize=1)
def table():
    """{question key: {"task", "subtask", "question", "models": {name: {...}}}} for questions every model answered."""
    path = cache_dir() / "table.json"
    reg = registry()["models"]
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("models") == sorted(reg):
            return cached["questions"]
    questions = {}
    jobs = [(name, cfg["release"], run) for name, cfg in reg.items() for run in runs_for(cfg["helm"], cfg["release"])]
    with ThreadPoolExecutor(8) as pool:
        loaded = list(pool.map(lambda job: (job, download(job[1], job[2])), jobs))
    for (name, release, run), (predictions, instances) in loaded:
        scenario = run.split(":")[0]
        task, metric = TASKS[scenario]
        text = {i["id"]: i["input"]["text"] for i in instances}
        for p in predictions:
            if p.get("train_trial_index", 0) != 0 or metric not in p["stats"]:
                continue
            sub = subtask_of(run)
            key = f"{task}/{sub}/{p['instance_id']}"
            q = questions.setdefault(key, {"task": task, "subtask": sub, "question": text.get(p["instance_id"], ""), "models": {}})
            q["models"][name] = {"correct": int(p["stats"][metric] >= 1.0), "prompt_tokens": int(p["stats"]["num_prompt_tokens"]),
                                 "output_tokens": int(p["stats"]["num_output_tokens"]), "answer": normalise(scenario, p)}
    complete = {k: q for k, q in questions.items() if len(q["models"]) == len(reg) and q["question"]}
    path.write_text(json.dumps({"models": sorted(reg), "questions": complete}))
    return complete


def model_dir(name) -> Path:
    url, sha, folder, _, _ = MODELS[name]
    target = cache_dir() / folder
    if not (target / "tokenizer.json").exists():
        archive = cache_dir() / f"{folder}.tar.gz"
        if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != sha:
            with urllib.request.urlopen(url, timeout=300) as r:
                body = r.read()
            if hashlib.sha256(body).hexdigest() != sha:
                raise RuntimeError(f"{name}: the download does not match its pinned sha256")
            archive.write_bytes(body)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if not Path(m.name).name.startswith("._")]
            tar.extractall(cache_dir(), members=members, filter="data")
    return target
