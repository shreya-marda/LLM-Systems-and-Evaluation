from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


URL = "http://localhost:8000/generate"
CONCURRENCY_LEVELS = [1, 4, 8]

SHORT_PROMPTS = [
    "Summarize attention in one sentence.",
    "Give one reason prompt caching helps.",
    "Name a deterministic decoding strategy.",
]

LONG_PROMPTS = [
    (
        "You are evaluating a lightweight language model used in an interview assignment. "
        "Explain how a multiple-choice benchmark should be prompted so that the model returns "
        "a clean answer letter instead of a paragraph. Discuss the risks of ambiguous output "
        "formats, why deterministic settings matter for reproducibility, and how a short "
        "instruction can improve answer extraction without changing the model weights. End with "
        "a concise recommendation for practical evaluation on a local FastAPI generation server."
    ),
    (
        "Write a detailed explanation of why latency measurements for text generation should be "
        "broken into time to first token, total latency, and estimated tokens per second. Include "
        "the effect of concurrency, prompt length, decoding strategy, and server implementation "
        "details. Assume the system is a small Transformers model behind a local HTTP endpoint and "
        "the evaluator wants measurements that are simple, reproducible, and useful for comparing "
        "baseline and optimized prompting strategies across repeated runs."
    ),
    (
        "Imagine you are reviewing an LLM evaluation pipeline with a custom lm-eval wrapper, a "
        "guardrails module, a prompt optimization stage, and a load-testing script. Describe how "
        "these components fit together in a realistic engineering workflow. Cover dataset "
        "preparation, output validation, deterministic checks, benchmark reporting, and the tradeoff "
        "between faster approximate testing and slower high-confidence evaluation. Keep the response "
        "structured and informative enough to stress a local text generation endpoint with a long prompt."
    ),
]


def post_generate(prompt: str, timeout: float = 120.0) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "max_tokens": 80,
        "temperature": 0.2,
        "top_p": 0.95,
        "stop": None,
    }

    started = time.perf_counter()
    response = requests.post(URL, json=payload, stream=True, timeout=timeout)
    response.raise_for_status()
    first_chunk_time: float | None = None
    raw_body: list[bytes] = []

    for chunk in response.iter_content(chunk_size=128):
        if not chunk:
            continue
        raw_body.append(chunk)
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()

    finished = time.perf_counter()
    response.close()

    ttft_ms = ((first_chunk_time or finished) - started) * 1000.0
    total_latency_ms = (finished - started) * 1000.0

    response_data = response.json() if not raw_body else None
    if response_data is None:
        body = b"".join(raw_body).decode("utf-8")
        response_data = requests.models.complexjson.loads(body)

    text = response_data.get("text", "")
    if not isinstance(text, str):
        raise RuntimeError("Server response did not include a string `text` field.")

    word_count = max(len(text.split()), 1)
    total_latency_s = max(total_latency_ms / 1000.0, 1e-9)
    tpot = word_count / total_latency_s

    return {
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "tpot": tpot,
        "text": text,
    }


def run_scenario(prompt_type: str, prompts: list[str], concurrency: int) -> list[dict[str, Any]]:
    scenario_name = f"{prompt_type}_c{concurrency}"
    scheduled_prompts = [prompts[index % len(prompts)] for index in range(concurrency)]
    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(post_generate, prompt): prompt
            for prompt in scheduled_prompts
        }
        for future in as_completed(futures):
            result = future.result()
            rows.append(
                {
                    "scenario": scenario_name,
                    "concurrency": concurrency,
                    "prompt_type": prompt_type,
                    "ttft_ms": result["ttft_ms"],
                    "tpot": result["tpot"],
                    "total_latency_ms": result["total_latency_ms"],
                }
            )

    return rows


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []

    for (prompt_type, concurrency), group in df.groupby(["prompt_type", "concurrency"]):
        latency_values = group["total_latency_ms"].to_numpy()
        summaries.append(
            {
                "prompt_type": prompt_type,
                "concurrency": concurrency,
                "requests": len(group),
                "ttft_mean_ms": group["ttft_ms"].mean(),
                "tpot_mean": group["tpot"].mean(),
                "latency_p50_ms": np.percentile(latency_values, 50),
                "latency_p95_ms": np.percentile(latency_values, 95),
                "latency_p99_ms": np.percentile(latency_values, 99),
            }
        )

    return pd.DataFrame(summaries).sort_values(["prompt_type", "concurrency"])


def save_metrics(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "concurrency",
                "prompt_type",
                "ttft_ms",
                "tpot",
                "total_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows: list[dict[str, Any]] = []

    for concurrency in CONCURRENCY_LEVELS:
        all_rows.extend(run_scenario("short", SHORT_PROMPTS, concurrency))
        all_rows.extend(run_scenario("long", LONG_PROMPTS, concurrency))

    output_path = Path(__file__).resolve().parent / "metrics.csv"
    save_metrics(all_rows, output_path)

    df = pd.DataFrame(all_rows)
    summary = build_summary_table(df)

    print("Load test summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print()
    print(f"Saved detailed metrics to {output_path}")


if __name__ == "__main__":
    main()
