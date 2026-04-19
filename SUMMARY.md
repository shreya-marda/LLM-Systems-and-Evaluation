# Summary: Best Improvement and What I Learned

## The Story

The assignment asked for inference-time improvements to HellaSwag without finetuning.
The model was `distilgpt2` — a 82M parameter model served through a Transformers + FastAPI
backend on CPU, substituting for vLLM which does not support Windows natively.

### What Was Tried

Three techniques were stacked on top of a zero-shot greedy baseline:

1. **Strict prompt + stop sequences** — wrapping every prompt with an instruction to reply
   with exactly one capital letter, and stopping generation at newlines or explanation markers.
   This directly attacked distilgpt2's tendency to ramble instead of answering.

2. **Dynamic few-shot selection** — for each test example, the three most semantically similar
   calibration examples were retrieved using cosine similarity over sentence-transformer embeddings
   (all-MiniLM-L6-v2). This gave the model relevant context without manual curation.

3. **Majority voting (self-consistency)** — three samples were drawn at temperature=0.3 and
   the most common answer letter was selected. This reduced variance in the optimized pass.

### What Failed

Chain-of-thought prompting added no measurable gain on its own. distilgpt2 is too small to
follow reasoning instructions — it ignores the "think step by step" directive and produces
the same outputs regardless. CoT contributed zero estimated delta in the ablation table.

### What Worked

The single biggest lever was **output normalization**. The baseline frequently returned `None`
because distilgpt2 did not produce a clean A/B/C/D letter. A cascading extractor — trying
regex patterns from strict to loose, then falling back to token overlap with the actual
endings — rescued three of the seven flipped-to-correct examples. These were not smarter
predictions; they were the same predictions extracted more reliably.

### Results

| Pass | Accuracy | 95% Wilson CI |
| --- | ---: | ---: |
| Baseline | 0.1800 | [0.0977, 0.3080] |
| Optimized | 0.2200 | [0.1275, 0.3524] |

Lift: **+4.0 percentage points**. CIs overlap, so this is directional rather than
statistically significant at n=50. A run on n=500+ would be needed for p < 0.05.

### Key Insight

For a model as weak as distilgpt2, **reliable answer extraction matters more than prompt
sophistication**. Before chasing few-shot strategies or chain-of-thought, make sure the
model's output is being parsed correctly. Half the improvement came from not throwing away
valid answers that were buried in the response text.

## What This Taught Me

This assignment reinforced that real ML engineering is mostly about making good decisions under constraints. I had to adapt quickly when vLLM was not workable on Windows, and that clarified an important lesson for me: the serving backend matters less than preserving the interface contract that everything else depends on. Once the API was stable, the rest of the pipeline became an infrastructure problem: evaluation, caching, guardrails, performance measurement, and reporting all built on top of that contract.

The biggest modeling insight was that with a small model like `distilgpt2`, output handling mattered more than sophisticated prompting. A meaningful part of the lift came from better answer extraction and stricter formatting, not from the model suddenly reasoning better. I also saw how inference-time tricks like majority voting can help, but only with real variance and latency costs, which makes careful measurement and honest confidence intervals essential.

Overall, my takeaway is simple: build the right interface, parse outputs reliably, stay honest about uncertainty, and document every important constraint clearly. In practice, those engineering choices often matter more than the model itself.
