"""python -m autopilot.demo   write the live demo's data (docs/data.json) from results/bench.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def build(out=ROOT / "docs"):
    r = json.loads((RESULTS / "bench.json").read_text())
    out.mkdir(exist_ok=True)
    (out / "data.json").write_text(json.dumps({k: r[k] for k in ("questions", "tasks", "prices", "registries")}, indent=1))
    print(f"wrote {out / 'data.json'}")


if __name__ == "__main__":
    build()
