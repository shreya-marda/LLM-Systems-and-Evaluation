from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import requests

try:
    from improve.optimize_prompt import build_prompt, get_few_shots, majority_vote
except ImportError:
    from optimize_prompt import build_prompt, get_few_shots, majority_vote


DEFAULT_URL = "http://localhost:8000/generate"
N_VOTE_SAMPLES = 3
N_EXAMPLES_TO_SAVE = 15
TEST_LIMIT = 50  # reduced from 300 for speed

ANSWER_ONLY_INSTRUCTION = (
    "Choose the single best ending.\n"
    "Reply with exactly one capital letter: A, B, C, or D.\n"
    "Do not explain your answer.\n\n"
)
GEN_STOP = ["\n", "Explanation:", "Context:", "Q:"]


def load_inputs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    improve_dir = root / "improve"
    splits = json.loads((improve_dir / "hellaswag_splits.json").read_text(encoding="utf-8"))
    calibration_examples = splits["calibration"]
    test_examples = splits["test"][:TEST_LIMIT]  # limit for speed
    cal_embeddings = np.load(improve_dir / "cal_embeddings.npy")
    test_embeddings = np.load(improve_dir / "test_embeddings.npy")[:TEST_LIMIT]
    return calibration_examples, test_examples, cal_embeddings, test_embeddings


# --- answer extraction helpers ---

def extract_answer_letter(text: str) -> str | None:
    patterns = [
        r"^\s*([ABCD])\s*$",
        r"answer\s*[:\-]?\s*([ABCD])\b",
        r"the answer is\s*([ABCD])\b",
        r"\boption\s*([ABCD])\b",
        r"\bchoice\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]
    upper = text.upper()
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def normalize_prediction(text: str, example: dict[str, Any]) -> str | None:
    # try letter extraction first
    letter = extract_answer_letter(text)
    if letter is not None:
        return letter

    # fall back to matching ending text
    normalized_response = _normalize_text(text)
    if not normalized_response:
        return None

    endings = example.get("endings") or []
    if len(endings) != 4:
        return None

    scores: list[int] = []
    for ending in endings:
        normalized_ending = _normalize_text(ending)
        if not normalized_ending:
            scores.append(0)
            continue
        if normalized_ending in normalized_response:
            scores.append(len(normalized_ending))
            continue
        response_tokens = set(normalized_response.split())
        ending_tokens = set(normalized_ending.split())
        scores.append(len(response_tokens & ending_tokens))

    best_index = int(np.argmax(scores))
    if scores[best_index] > 0:
        return ["A", "B", "C", "D"][best_index]
    return None


def make_strict_prompt(
    example: dict[str, Any],
    *,
    few_shots: list[dict[str, Any]] | None = None,
    use_cot: bool = False,
) -> str:
    return ANSWER_ONLY_INSTRUCTION + build_prompt(example, few_shots=few_shots, use_cot=use_cot)


def gold_letter(example: dict[str, Any]) -> str:
    label = example.get("label")
    if isinstance(label, str) and label.strip().isdigit():
        label = int(label)
    if not isinstance(label, int) or not 0 <= label <= 3:
        raise ValueError(f"Example has invalid label: {label!r}")
    return ["A", "B", "C", "D"][label]


def generate(
    prompt: str,
    *,
    url: str = DEFAULT_URL,
    max_tokens: int = 50,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    timeout: float = 120.0,
) -> str:
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    text = data.get("text")
    if not isinstance(text, str):
        raise RuntimeError("Server response did not include a string `text` field.")
    return text


def wilson_score_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denominator = 1 + (z * z) / total
    center = (phat + (z * z) / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1 - phat) / total) + ((z * z) / (4 * total * total)))
        / denominator
    )
    return center - margin, center + margin


def evaluate_baseline(test_examples: list[dict[str, Any]], *, url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, example in enumerate(test_examples):
        prompt = make_strict_prompt(example, few_shots=None, use_cot=False)
        response = generate(
            prompt,
            url=url,
            max_tokens=6,
            temperature=0.0,
            top_p=1.0,
            stop=GEN_STOP,
        )
        prediction = normalize_prediction(response, example)
        gold = gold_letter(example)
        rows.append({
            "example_id": example.get("ind"),
            "gold": gold,
            "prediction": prediction,
            "correct": prediction == gold,
            "prompt": prompt,
            "response": response,
        })
        if (i + 1) % 10 == 0:
            correct_so_far = sum(1 for r in rows if r["correct"])
            print(f"  Baseline {i+1}/{len(test_examples)} | acc so far: {correct_so_far/(i+1):.3f}")
    return rows


def evaluate_optimized(
    test_examples: list[dict[str, Any]],
    calibration_examples: list[dict[str, Any]],
    cal_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    *,
    url: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for i, (example, test_embedding) in enumerate(zip(test_examples, test_embeddings, strict=True)):
        few_shots = get_few_shots(
            example,
            calibration_examples,
            cal_embeddings,
            test_embedding,
            k=3,
        )
        prompt = make_strict_prompt(example, few_shots=few_shots, use_cot=True)
        responses = [
            generate(
                prompt,
                url=url,
                max_tokens=12,
                temperature=0.3,
                top_p=1.0,
                stop=GEN_STOP,
            )
            for _ in range(N_VOTE_SAMPLES)
        ]
        normalized_votes = [normalize_prediction(r, example) for r in responses]
        prediction = majority_vote([v for v in normalized_votes if v] or responses)
        gold = gold_letter(example)
        rows.append({
            "example_id": example.get("ind"),
            "gold": gold,
            "prediction": prediction,
            "correct": prediction == gold,
            "prompt": prompt,
            "responses": responses,
            "few_shot_ids": [shot.get("ind") for shot in few_shots],
        })
        if (i + 1) % 10 == 0:
            correct_so_far = sum(1 for r in rows if r["correct"])
            print(f"  Optimized {i+1}/{len(test_examples)} | acc so far: {correct_so_far/(i+1):.3f}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    accuracy = (correct / total) if total else 0.0
    ci_low, ci_high = wilson_score_interval(correct, total)
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "ci_95": {"low": ci_low, "high": ci_high},
    }


def select_examples_to_save(
    test_examples: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    optimized_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for example, baseline, optimized in zip(test_examples, baseline_rows, optimized_rows, strict=True):
        record = {
            "example_id": example.get("ind"),
            "ctx": example.get("ctx"),
            "endings": example.get("endings"),
            "gold": gold_letter(example),
            "baseline_prediction": baseline["prediction"],
            "baseline_correct": baseline["correct"],
            "baseline_response": baseline["response"],
            "optimized_prediction": optimized["prediction"],
            "optimized_correct": optimized["correct"],
            "optimized_responses": optimized["responses"],
            "few_shot_ids": optimized["few_shot_ids"],
            "flipped_to_correct": (not baseline["correct"]) and optimized["correct"],
            "flipped_to_incorrect": baseline["correct"] and (not optimized["correct"]),
        }
        enriched.append(record)

    flipped = [r for r in enriched if r["flipped_to_correct"] or r["flipped_to_incorrect"]]
    improved = [r for r in flipped if r["flipped_to_correct"]]
    regressed = [r for r in flipped if r["flipped_to_incorrect"]]
    selected = improved + regressed
    if len(selected) < N_EXAMPLES_TO_SAVE:
        extras = [r for r in enriched if r["example_id"] not in {s["example_id"] for s in selected}]
        selected.extend(extras[: N_EXAMPLES_TO_SAVE - len(selected)])
    return selected[:N_EXAMPLES_TO_SAVE]


def print_comparison_table(baseline_summary: dict[str, Any], optimized_summary: dict[str, Any]) -> None:
    print("\n=== Results ===")
    print("| pass | accuracy | 95% wilson ci |")
    print("| --- | ---: | ---: |")
    print(
        f"| Baseline  | {baseline_summary['accuracy']:.4f} | "
        f"[{baseline_summary['ci_95']['low']:.4f}, {baseline_summary['ci_95']['high']:.4f}] |"
    )
    print(
        f"| Optimized | {optimized_summary['accuracy']:.4f} | "
        f"[{optimized_summary['ci_95']['low']:.4f}, {optimized_summary['ci_95']['high']:.4f}] |"
    )
    lift = optimized_summary['accuracy'] - baseline_summary['accuracy']
    print(f"\nLift: {lift:+.4f} ({lift*100:+.2f} percentage points)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading data...")
    calibration_examples, test_examples, cal_embeddings, test_embeddings = load_inputs(root)
    print(f"Running on {len(test_examples)} test examples\n")

    print("=== Running Baseline ===")
    baseline_rows = evaluate_baseline(test_examples, url=DEFAULT_URL)

    print("\n=== Running Optimized ===")
    optimized_rows = evaluate_optimized(
        test_examples, calibration_examples, cal_embeddings, test_embeddings, url=DEFAULT_URL
    )

    baseline_summary = summarize(baseline_rows)
    optimized_summary = summarize(optimized_rows)

    examples_to_save = select_examples_to_save(test_examples, baseline_rows, optimized_rows)
    improve_examples_path = results_dir / "improve_examples.json"
    improve_examples_path.write_text(json.dumps(examples_to_save, indent=2), encoding="utf-8")

    output = {
        "dataset": "hellaswag",
        "test_limit": TEST_LIMIT,
        "server_url": DEFAULT_URL,
        "baseline": {"description": "zero-shot, temperature=0, greedy, max_tokens=6, stop sequences", **baseline_summary},
        "optimized": {"description": "3-shot dynamic few-shot + CoT + majority vote k=3, temperature=0.3, max_tokens=12, stop sequences", **optimized_summary},
        "saved_examples_path": str(improve_examples_path),
    }
    improve_results_path = results_dir / "improve_results.json"
    improve_results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print_comparison_table(baseline_summary, optimized_summary)
    print(f"\nSaved examples  -> {improve_examples_path}")
    print(f"Saved results   -> {improve_results_path}")


if __name__ == "__main__":
    main()