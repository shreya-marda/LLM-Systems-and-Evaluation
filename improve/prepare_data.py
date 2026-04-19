from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


DATASET_NAME = "allenai/hellaswag"
DATASET_SPLIT = "validation"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CALIBRATION_SIZE = 200
TEST_SIZE = 300


def select_splits(dataset_split) -> tuple[list[dict], list[dict]]:
    required_examples = CALIBRATION_SIZE + TEST_SIZE
    if len(dataset_split) < required_examples:
        raise ValueError(
            f"Need at least {required_examples} rows from {DATASET_NAME}/{DATASET_SPLIT}, "
            f"but found only {len(dataset_split)}."
        )

    calibration = [dataset_split[i] for i in range(CALIBRATION_SIZE)]
    test = [dataset_split[i] for i in range(CALIBRATION_SIZE, required_examples)]
    return calibration, test


def extract_ctx_texts(rows: list[dict]) -> list[str]:
    texts: list[str] = []
    for index, row in enumerate(rows, start=1):
        text = row.get("ctx")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Row {index} is missing a usable `ctx` field.")
        texts.append(text)
    return texts


def serialize_rows(rows: list[dict]) -> list[dict]:
    serialized: list[dict] = []
    for row in rows:
        serialized.append(
            {
                "ind": row.get("ind"),
                "activity_label": row.get("activity_label"),
                "ctx_a": row.get("ctx_a"),
                "ctx_b": row.get("ctx_b"),
                "ctx": row.get("ctx"),
                "endings": row.get("endings"),
                "label": row.get("label"),
                "source_id": row.get("source_id"),
                "split": row.get("split"),
                "split_type": row.get("split_type"),
            }
        )
    return serialized


def main() -> None:
    root = Path(__file__).resolve().parent
    cal_embeddings_path = root / "cal_embeddings.npy"
    test_embeddings_path = root / "test_embeddings.npy"
    splits_path = root / "hellaswag_splits.json"

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    calibration_rows, test_rows = select_splits(dataset)

    model = SentenceTransformer(EMBEDDING_MODEL)
    calibration_embeddings = model.encode(
        extract_ctx_texts(calibration_rows),
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    test_embeddings = model.encode(
        extract_ctx_texts(test_rows),
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    np.save(cal_embeddings_path, calibration_embeddings)
    np.save(test_embeddings_path, test_embeddings)

    splits_payload = {
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "embedding_model": EMBEDDING_MODEL,
        "calibration_size": CALIBRATION_SIZE,
        "test_size": TEST_SIZE,
        "calibration": serialize_rows(calibration_rows),
        "test": serialize_rows(test_rows),
    }
    splits_path.write_text(json.dumps(splits_payload, indent=2), encoding="utf-8")

    print(f"Saved calibration embeddings to {cal_embeddings_path}")
    print(f"Saved test embeddings to {test_embeddings_path}")
    print(f"Saved split metadata to {splits_path}")


if __name__ == "__main__":
    main()
