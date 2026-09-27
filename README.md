# LLM Cost Autopilot

[![CI](https://github.com/umer-78/llm-cost-autopilot/actions/workflows/ci.yml/badge.svg)](https://github.com/umer-78/llm-cost-autopilot/actions/workflows/ci.yml)

A router that sends each LLM request to the cheapest model likely to get it right. Measured on
4,551 questions whose answers from every model were recorded by
[HELM Lite](https://crfm.stanford.edu/helm/lite/) (GSM8K, MATH, MMLU, MedQA, OpenBookQA,
LegalBench), it **matched GPT-4o's accuracy at 23% of the cost** with US-hosted models, and at
**10.5% of the cost** when DeepSeek-V3 is allowed.

**Live demo:** https://umer-78.github.io/llm-cost-autopilot/ (every strategy's cost and accuracy, and the router's confidence as a slider)

| Strategy (US providers) | Accuracy | $ per 1,000 requests | Saving |
|---|---|---|---|
| everything to GPT-4o | 77.6% | 3.64 | - |
| **learned router** | **77.8%** | **0.83** | **77.3%** |
| cascade: Gemini Flash + Llama 70B, GPT-4o when they disagree | 77.6% | 2.33 | 35.9% |
| fixed task-to-model map | 78.5% | 2.48 | 31.9% |
| oracle (cheapest model that was right) | 88.7% | 0.44 | 88.0% |

With DeepSeek-V3 in the registry the router reaches 78.3% at $0.38 (89.5% saving); DeepSeek-V3
alone gets 78.6% at $0.33. Router vs GPT-4o on the same held-out questions: +0.3 points (95%
interval -1.0 to +1.6), i.e. quality parity. Full tables: [`results/bench.md`](results/bench.md).

## How it works

- **Data** (`autopilot/data.py`): HELM's per-question answers, correctness and token counts for
  six models, downloaded on first run; questions split in half by a hash of their id.
- **Router** (`autopilot/router.py`): one logistic regression per model predicts P(correct) from
  the prompt's bge-small embedding, its task and its length. It routes to the cheapest model whose
  chance clears a confidence level, else the model it trusts most. The confidence is tuned on
  out-of-fold predictions over the training half to stay within half a point of GPT-4o.
- **Prices** (`autopilot/models.yaml`): providers' list prices (early 2025) × HELM's token counts.
- **Service** (`autopilot/service.py`): `POST /v1/chat/completions` (the router picks the model and
  reports it in `X-Routed-Model`), `GET /v1/models`, `GET /v1/stats`, `PUT /v1/routing-config`.

Per-question routing beats a per-task map because models fail on different questions within a
task: Gemini 1.5 Flash gets 93% of MATH level 1 but 33% of GSM8K, and the router learns which.

## Run

```bash
pip install -e '.[dev]'
pytest -q                  # 4 tests
python -m autopilot bench  # downloads HELM results (~65 MB) and prints the tables
python -m autopilot.demo   # rebuild the live demo's data in docs/
python -m autopilot gate   # CI: router within a point of GPT-4o, saving within 5 points of baseline
```

## Limits

- HELM prompts are benchmark questions with few-shot examples, not production traffic.
- Dollar figures are list prices; only their ratios affect routing.
- A cheap router still needs labels: here they come from HELM; in production, from a verifier
  that re-asks the frontier model on a sample.

MIT licence. HELM results and the underlying datasets keep their own licences and are not
redistributed.
