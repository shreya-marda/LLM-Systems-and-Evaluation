from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lm_eval import simple_evaluate

try:
    from eval_runner.vllm_model import LocalFastAPIGenerateLM
except ImportError:
    from vllm_model import LocalFastAPIGenerateLM


DEFAULT_TASKS = ["hellaswag", "mmlu_abstract_algebra"]
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_LIMIT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness benchmarks against the local FastAPI model wrapper."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the local FastAPI generation server.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of examples per task to evaluate.",
    )
    return parser.parse_args()


def build_model(base_url: str) -> LocalFastAPIGenerateLM:
    return LocalFastAPIGenerateLM(
        base_url=base_url,
        batch_size=1,
        max_gen_toks=128,
        temperature=0.0,
        top_p=1.0,
    )


def run_harness_tasks(
    model: LocalFastAPIGenerateLM,
    limit: int,
) -> dict[str, Any]:
    return simple_evaluate(
        model=model,
        tasks=DEFAULT_TASKS,
        num_fewshot=0,
        limit=limit,
        batch_size=1,
    )


def load_custom_examples(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Custom task file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Custom task JSON must be a list of objects.")

    examples: list[dict[str, str]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Custom task item {index} is not an object.")
        prompt = item.get("prompt")
        target = item.get("target")
        if not isinstance(prompt, str) or not isinstance(target, str):
            raise ValueError(f"Custom task item {index} must include string prompt and target fields.")
        examples.append({"prompt": prompt, "target": target})

    return examples[:limit]


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def run_custom_task(
    model: LocalFastAPIGenerateLM,
    custom_task_path: Path,
    limit: int,
) -> dict[str, Any]:
    examples = load_custom_examples(custom_task_path, limit=limit)
    predictions: list[dict[str, Any]] = []
    correct = 0

    for example in examples:
        prediction = model.generate_until(
            [(example["prompt"], {"max_gen_toks": 32, "temperature": 0.0, "top_p": 1.0})]
        )[0]
        is_correct = normalize_text(prediction).startswith(normalize_text(example["target"]))
        correct += int(is_correct)
        predictions.append(
            {
                "prompt": example["prompt"],
                "target": example["target"],
                "prediction": prediction,
                "correct": is_correct,
            }
        )

    total = len(examples)
    accuracy = (correct / total) if total else 0.0
    return {
        "task_name": "custom_task",
        "num_fewshot": 0,
        "limit": limit,
        "samples_evaluated": total,
        "accuracy": accuracy,
        "samples": predictions,
    }


def extract_accuracy(task_results: dict[str, Any]) -> float | None:
    for key, value in task_results.items():
        if key.endswith("_stderr,none"):
            continue
        if key in {"acc,none", "acc_norm,none", "exact_match,none"} and isinstance(value, (int, float)):
            return float(value)
        if key.startswith("acc") and isinstance(value, (int, float)):
            return float(value)
        if key.startswith("exact_match") and isinstance(value, (int, float)):
            return float(value)
    return None


def build_summary_rows(
    harness_results: dict[str, Any],
    custom_results: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_results = harness_results.get("results", {})

    for task_name in DEFAULT_TASKS:
        metrics = task_results.get(task_name, {})
        rows.append(
            {
                "task_name": task_name,
                "num_shots": 0,
                "accuracy": extract_accuracy(metrics),
            }
        )

    rows.append(
        {
            "task_name": custom_results["task_name"],
            "num_shots": custom_results["num_fewshot"],
            "accuracy": custom_results["accuracy"],
        }
    )
    return rows


def print_markdown_summary(rows: list[dict[str, Any]]) -> None:
    print("| task name | num_shots | accuracy |")
    print("| --- | ---: | ---: |")
    for row in rows:
        accuracy = row["accuracy"]
        accuracy_str = "n/a" if accuracy is None else f"{accuracy:.4f}"
        print(f"| {row['task_name']} | {row['num_shots']} | {accuracy_str} |")


def build_output(
    base_url: str,
    limit: int,
    harness_results: dict[str, Any],
    custom_results: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "base_url": base_url,
        "limit": limit,
        "tasks": DEFAULT_TASKS + [custom_results["task_name"]],
        "summary": summary_rows,
        "harness_results": harness_results,
        "custom_task_results": custom_results,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    custom_task_path = root / "eval_runner" / "custom_task" / "custom_task.json"

    model = build_model(args.base_url)
    harness_results = run_harness_tasks(model, limit=args.limit)
    custom_results = run_custom_task(model, custom_task_path=custom_task_path, limit=args.limit)
    summary_rows = build_summary_rows(harness_results, custom_results)

    print_markdown_summary(summary_rows)

    output = build_output(
        base_url=args.base_url,
        limit=args.limit,
        harness_results=harness_results,
        custom_results=custom_results,
        summary_rows=summary_rows,
    )
    output_path = results_dir / "raw_results.json"
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Saved full results to {output_path}")


if __name__ == "__main__":
    main()
