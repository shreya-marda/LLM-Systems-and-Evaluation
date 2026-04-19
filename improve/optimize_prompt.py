from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np


LETTER_CHOICES = ["A", "B", "C", "D"]


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _answer_letter(example: dict[str, Any]) -> str | None:
    label = example.get("label")

    if isinstance(label, str) and label.strip().isdigit():
        label = int(label)

    if isinstance(label, int) and 0 <= label < len(LETTER_CHOICES):
        return LETTER_CHOICES[label]

    return None


def _format_example_block(example: dict[str, Any], include_answer: bool) -> str:
    question_text = example.get("ctx") or ""
    endings = example.get("endings") or []
    if len(endings) != 4:
        raise ValueError("Expected a HellaSwag example with exactly 4 endings.")

    lines = [
        f"Context: {question_text}",
        f"A. {endings[0]}",
        f"B. {endings[1]}",
        f"C. {endings[2]}",
        f"D. {endings[3]}",
    ]

    if include_answer:
        answer = _answer_letter(example)
        if answer is None:
            raise ValueError("Few-shot examples must include a valid label in the range 0-3.")
        lines.append(f"Answer: {answer}")
    else:
        lines.append("Answer:")

    return "\n".join(lines)


def get_few_shots(
    test_example: dict[str, Any],
    cal_examples: list[dict[str, Any]],
    cal_embeddings: np.ndarray,
    test_embedding: np.ndarray,
    k: int = 3,
) -> list[dict[str, Any]]:
    if k <= 0:
        return []

    if len(cal_examples) != len(cal_embeddings):
        raise ValueError("Calibration examples and calibration embeddings must have the same length.")

    test_vector = _normalize_embedding(np.asarray(test_embedding))
    cal_matrix = np.asarray(cal_embeddings, dtype=float)
    cal_norms = np.linalg.norm(cal_matrix, axis=1, keepdims=True)
    cal_norms[cal_norms == 0] = 1.0
    cal_matrix = cal_matrix / cal_norms

    similarities = cal_matrix @ test_vector
    ranked_indices = np.argsort(similarities)[::-1]

    selected: list[dict[str, Any]] = []
    test_id = test_example.get("ind")

    for index in ranked_indices:
        candidate = cal_examples[int(index)]
        if test_id is not None and candidate.get("ind") == test_id:
            continue
        selected.append(candidate)
        if len(selected) == min(k, len(cal_examples)):
            break

    return selected


def build_prompt(
    example: dict[str, Any],
    few_shots: list[dict[str, Any]] | None = None,
    use_cot: bool = False,
) -> str:
    sections: list[str] = []

    if use_cot:
        sections.append("Think step by step then pick the best ending: A, B, C, or D.")

    if few_shots:
        for shot in few_shots:
            sections.append(_format_example_block(shot, include_answer=True))

    sections.append(_format_example_block(example, include_answer=False))
    return "\n\n".join(sections)


def majority_vote(responses: list[str]) -> str | None:
    extracted: list[str] = []
    for response in responses:
        match = re.search(r"\b([ABCD])\b", response.upper())
        if match:
            extracted.append(match.group(1))

    if not extracted:
        return None

    counts = Counter(extracted)
    return counts.most_common(1)[0][0]
