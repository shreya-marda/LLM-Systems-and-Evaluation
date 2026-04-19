from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests


DEFAULT_URL = "http://localhost:8000/generate"


def generate(
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
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


def stream_generate(
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stop: list[str] | None = None,
    url: str = DEFAULT_URL,
    timeout: float = 120.0,
    word_delay: float = 0.08,
) -> str:
    text = generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        url=url,
        timeout=timeout,
    )

    words = text.split()
    for index, word in enumerate(words):
        suffix = "" if index == len(words) - 1 else " "
        print(word, end=suffix, flush=True)
        time.sleep(word_delay)
    print()

    return text


def run_concurrent(
    prompts: list[str],
    n_workers: int = 4,
    *,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stop: list[str] | None = None,
    url: str = DEFAULT_URL,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    def worker(prompt: str) -> dict[str, Any]:
        started = time.perf_counter()
        text = generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            url=url,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
        return {
            "prompt": prompt,
            "text": text,
            "elapsed_s": elapsed,
        }

    results: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker, prompt): prompt for prompt in prompts}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"[{index}/{len(prompts)}] {result['elapsed_s']:.2f}s | {result['prompt']}")

    total_elapsed = time.perf_counter() - overall_start
    print(f"Completed {len(prompts)} requests in {total_elapsed:.2f}s using {n_workers} workers")
    return results


def main() -> None:
    sample_prompts = [
        "Explain what an LLM evaluation pipeline does in one sentence.",
        "Write two short bullet points about prompt quality.",
        "Give one reason to cache repeated evaluation requests.",
    ]

    print("=== Sequential samples ===")
    for index, prompt in enumerate(sample_prompts, start=1):
        started = time.perf_counter()
        text = generate(prompt, max_tokens=64, temperature=0.3, top_p=0.9)
        elapsed = time.perf_counter() - started
        print(f"Sample {index}: {prompt}")
        print(text)
        print(f"Completed in {elapsed:.2f}s")
        print()

    print("=== Simulated streaming ===")
    stream_generate(sample_prompts[0], max_tokens=64, temperature=0.3, top_p=0.9)
    print()

    print("=== Concurrent run ===")
    results = run_concurrent(sample_prompts, n_workers=3, max_tokens=64, temperature=0.3, top_p=0.9)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
