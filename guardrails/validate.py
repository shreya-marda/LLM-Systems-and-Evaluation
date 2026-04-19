from __future__ import annotations

import difflib
import json
import re
from typing import Any

import requests


DEFAULT_URL = "http://localhost:8000/generate"
DEFAULT_PROMPT = (
    "Answer the following multiple choice question with only the letter.\n"
    "Which decoding method is deterministic when temperature is zero?\n"
    "A. Top-p sampling\n"
    "B. Beam search with random tie breaking\n"
    "C. Greedy decoding\n"
    "D. Temperature annealing"
)


def generate(
    prompt: str,
    *,
    max_tokens: int = 50,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    url: str = DEFAULT_URL,
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


def determinism_check(
    prompt: str,
    *,
    n_runs: int = 5,
    url: str = DEFAULT_URL,
) -> tuple[bool, list[str]]:
    outputs = [
        generate(
            prompt,
            max_tokens=50,
            temperature=0.0,
            top_p=1.0,
            url=url,
        )
        for _ in range(n_runs)
    ]
    passed = len(set(outputs)) == 1
    return passed, outputs


def format_diff(reference: str, candidate: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            reference.splitlines(),
            candidate.splitlines(),
            fromfile="run_1",
            tofile="run_n",
            lineterm="",
        )
    )


def print_determinism_report(outputs: list[str]) -> None:
    reference = outputs[0]
    for index, output in enumerate(outputs[1:], start=2):
        if output != reference:
            print(f"FAIL: output {index} differed from output 1")
            diff = format_diff(reference, output)
            print(diff if diff else "(difference detected but no line-level diff available)")
            return

    print("PASS: all 5 deterministic generations matched exactly")


def validate_output(
    text: str,
    task_type: str,
    *,
    required_keys: list[str] | None = None,
) -> tuple[bool, str]:
    if task_type == "multiple_choice":
        if re.search(r"\b([ABCD])\b", text):
            return True, "Found a valid multiple-choice label (A, B, C, or D)."
        return False, "Did not find a standalone multiple-choice label A, B, C, or D."

    if task_type == "json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return False, f"Invalid JSON: {exc}"

        if required_keys:
            if not isinstance(parsed, dict):
                return False, "JSON output must be an object when required keys are specified."

            missing = [key for key in required_keys if key not in parsed]
            if missing:
                return False, f"Missing required keys: {', '.join(missing)}"

        return True, "Valid JSON output."

    raise ValueError(f"Unsupported task_type: {task_type}")


def main() -> None:
    print("=== Determinism check ===")
    passed, outputs = determinism_check(DEFAULT_PROMPT)
    print_determinism_report(outputs)
    if not passed:
        for index, output in enumerate(outputs, start=1):
            print()
            print(f"Run {index}:")
            print(output)

    print()
    print("=== Output validation demo ===")
    mc_text = outputs[0] if outputs else "C"
    mc_ok, mc_message = validate_output(mc_text, "multiple_choice")
    print(f"multiple_choice: {'PASS' if mc_ok else 'FAIL'} - {mc_message}")

    json_text = '{"label": "safe", "score": 0.98}'
    json_ok, json_message = validate_output(
        json_text,
        "json",
        required_keys=["label", "score"],
    )
    print(f"json: {'PASS' if json_ok else 'FAIL'} - {json_message}")


if __name__ == "__main__":
    main()
