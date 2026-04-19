# LLM Evaluation Pipeline — Mercor Interview Assignment

## Overview

A complete end-to-end LLM evaluation pipeline covering inference serving, benchmark
evaluation, performance profiling, output guardrails, and inference-time optimization.

**Backend note:** The assignment specifies vLLM serving. In this Windows-only environment,
vLLM installation failed during wheel build (vLLM has no Windows wheel). A backend-compatible
local inference server was implemented using Hugging Face Transformers with the same HTTP
endpoint contract. The `serve.py` interface is intentionally vLLM-compatible — swapping the
backend requires only changing the startup command, with zero changes to eval, perf,
guardrails, or improve layers.

---

## Project Structure

```
Mercor/
├── serve/
│   ├── serve.py              # FastAPI + Transformers inference server (distilgpt2)
│   └── client.py             # Python client: single, streaming, concurrent requests
├── eval_runner/
│   ├── vllm_model.py         # lm-eval LM subclass wrapping local /generate endpoint
│   ├── run_eval.py           # Benchmark runner: hellaswag, mmlu, custom task
│   └── custom_task/
│       ├── custom_task.json  # 20-question ML/AI multiple choice benchmark
│       └── custom_task.yaml  # lm-eval task config
├── perf/
│   ├── load_test.py          # Concurrent load generator, P50/P95/P99 latency
│   ├── metrics.csv           # Raw latency measurements
│   └── analysis.ipynb        # Charts and commentary
├── guardrails/
│   ├── validate.py           # Determinism check + output schema validation
│   └── README.md             # What was tested, where nondeterminism persists
├── improve/
│   ├── prepare_data.py       # HellaSwag splits + sentence embeddings
│   ├── optimize_prompt.py    # Few-shot retrieval, CoT, majority vote helpers
│   ├── infer.py              # Baseline vs optimized evaluation runner
│   └── report.md             # Results, ablation table, examples, config
├── results/
│   ├── raw_results.json      # lm-eval benchmark outputs
│   ├── improve_results.json  # Baseline vs optimized accuracy + CI
│   └── improve_examples.json # 15 real before/after examples
├── Makefile
├── README.md
└── SUMMARY.md
```

---

## Setup

```bash
pip install fastapi uvicorn transformers torch requests httpx pandas numpy \
            matplotlib jupyter lm-eval sentence-transformers datasets
```

---

## Usage

### Part A — Start the inference server

```bash
python -m uvicorn serve.serve:app --host 0.0.0.0 --port 8000
```

Test it:
```bash
python serve/client.py
```

The server exposes:
- `GET  /health`   — liveness check
- `POST /generate` — accepts `{prompt, max_tokens, temperature, top_p, stop}`, returns `{text}`

### Part B — Run benchmarks

```bash
python eval_runner/run_eval.py --limit 30
```

### Part C — Performance profiling

```bash
python perf/load_test.py
```

### Part D — Guardrails

```bash
python guardrails/validate.py
```

### Part E — Benchmark improvement

```bash
python improve/prepare_data.py   # one-time data prep
python improve/infer.py          # baseline vs optimized
```

---

## Results

### Benchmark Scores (limit=30, distilgpt2)

| Task | Shots | Accuracy |
| --- | ---: | ---: |
| hellaswag | 0 | 0.1000 |
| mmlu_abstract_algebra | 0 | 0.2000 |
| custom_task | 0 | 0.5000 |

### HellaSwag Improvement (n=50)

| Pass | Accuracy | 95% Wilson CI |
| --- | ---: | ---: |
| Baseline (zero-shot, greedy) | 0.1800 | [0.0977, 0.3080] |
| Optimized (3-shot + CoT + majority vote) | 0.2200 | [0.1275, 0.3524] |

**Lift: +4.0 percentage points — exceeds HellaSwag target of +3.0pp** via strict prompting, dynamic few-shot retrieval,
and majority voting with aggressive output normalization.

CIs overlap at n=50 — directional improvement, not statistically significant at p<0.05.
A run at n=500+ would be needed for a definitive claim.

### Performance (distilgpt2, CPU, Windows)

| Prompt type | Concurrency | P50 ms | P95 ms | P99 ms |
| --- | ---: | ---: | ---: | ---: |
| short | 1 | ~5,763 | ~5,763 | ~5,763 |
| short | 8 | ~25,055 | ~25,657 | ~25,660 |
| long | 1 | ~346 | ~346 | ~346 |
| long | 8 | ~35,414 | ~35,741 | ~35,759 |

Note: CPU inference has no batching benefit — latency scales linearly with concurrency.

---

## See Also

- `SUMMARY.md` — story of the best improvement and key insight
- `improve/report.md` — full ablation, 10 real examples, exact config
- `guardrails/README.md` — determinism testing notes


## RESULTS
<img width="1574" height="430" alt="image" src="https://github.com/user-attachments/assets/a26d481b-4091-435c-86db-1737301cf5bc" />
<img width="1583" height="319" alt="image" src="https://github.com/user-attachments/assets/1e60f399-2a03-410a-982f-9177ae20d179" />
<img width="1376" height="210" alt="image" src="https://github.com/user-attachments/assets/dc8caf25-f590-45d3-a45e-9fc4aa765f31" />
<img width="1545" height="784" alt="image" src="https://github.com/user-attachments/assets/f7623665-fb63-42de-bc0a-3c5fcec25932" />



