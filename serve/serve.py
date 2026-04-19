from __future__ import annotations

from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "distilgpt2"


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def apply_stop_sequences(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text

    cutoff = len(text)
    for sequence in stop:
        if not sequence:
            continue
        position = text.find(sequence)
        if position != -1:
            cutoff = min(cutoff, position)
    return text[:cutoff]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, str]:
    inputs = tokenizer(request.prompt, return_tensors="pt")

    generation_kwargs: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "max_new_tokens": request.max_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if request.temperature == 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = request.temperature
        generation_kwargs["top_p"] = request.top_p

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = apply_stop_sequences(text, request.stop)
    return {"text": text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
