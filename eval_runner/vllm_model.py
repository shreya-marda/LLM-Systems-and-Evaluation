from __future__ import annotations

import argparse
import hashlib
import json
import shelve
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("local-fastapi-generate")
class LocalFastAPIGenerateLM(LM):
    """
    lm-evaluation-harness wrapper for a local FastAPI text generation endpoint.

    Expected API:
      POST http://localhost:8000/generate
      Request: {"prompt": str, "max_tokens": int, "temperature": float, "top_p": float,
                "stop": list[str] | None}
      Response: {"text": str}

    Since this endpoint does not return token logprobs or echo the prompt, the
    loglikelihood-style methods use a coarse proxy score based on generated text length.
    """

    def __init__(
        self,
        model: str = "local-fastapi-generate",
        base_url: str = "http://localhost:8000",
        generate_path: str = "/generate",
        health_path: str = "/health",
        batch_size: int = 1,
        max_gen_toks: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: float = 120.0,
        cache_dir: str = "results/cache",
        cache_name: str = "local_fastapi_generate",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}{generate_path}"
        self.health_url = f"{self.base_url}{health_path}"
        self._batch_size = int(batch_size)
        self._max_gen_toks = int(max_gen_toks)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._timeout = float(timeout)
        self._extra_args = dict(kwargs)

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = str(self._cache_dir / cache_name)
        self._cache_lock = threading.Lock()

    @property
    def tokenizer_name(self) -> str:
        return f"{self.model}-fastapi-generate"

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def eot_token_id(self) -> int:
        return 0

    def tok_encode(self, string: str, **kwargs: Any) -> list[int]:
        raise NotImplementedError(
            "This remote wrapper does not expose local tokenization."
        )

    def tok_decode(self, tokens: list[int], **kwargs: Any) -> str:
        raise NotImplementedError(
            "This remote wrapper does not expose local token decoding."
        )

    @classmethod
    def create_from_arg_string(
        cls,
        arg_string: str,
        additional_config: dict[str, Any] | None = None,
    ) -> "LocalFastAPIGenerateLM":
        args = {}
        if arg_string:
            for item in arg_string.split(","):
                if not item.strip():
                    continue
                key, value = item.split("=", 1)
                args[key.strip()] = cls._coerce_arg(value.strip())

        if additional_config:
            args.update(additional_config)

        return cls(**args)

    @staticmethod
    def _coerce_arg(value: str) -> Any:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

        for caster in (int, float):
            try:
                return caster(value)
            except ValueError:
                pass

        if value.startswith("[") or value.startswith("{"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        return value

    def _cache_key(self, method: str, payload: dict[str, Any]) -> str:
        normalized = json.dumps(
            {"method": method, "payload": payload},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Any | None:
        with self._cache_lock:
            with shelve.open(self._cache_path) as db:
                return db.get(key)

    def _cache_set(self, key: str, value: Any) -> None:
        with self._cache_lock:
            with shelve.open(self._cache_path) as db:
                db[key] = value

    def _request(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"FastAPI request failed with HTTP {exc.code} at {url}: {details}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach server at {url}: {exc.reason}") from exc

    def _cached_generate(self, payload: dict[str, Any], cache_method: str) -> dict[str, Any]:
        key = self._cache_key(cache_method, payload)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        response = self._request(self.generate_url, method="POST", payload=payload)
        self._cache_set(key, response)
        return response

    def _build_payload(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        payload.update(self._extra_args)
        return payload

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        text = response.get("text")
        if not isinstance(text, str):
            raise RuntimeError("Generation response did not include a string `text` field.")
        return text

    def _proxy_loglikelihood(self, context: str, continuation: str) -> tuple[float, bool]:
        prompt = f"{context}{continuation}"
        payload = self._build_payload(
            prompt,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )
        response = self._cached_generate(payload, cache_method="loglikelihood")
        generated = self._extract_text(response)

        score = -float(len(generated))
        is_greedy = len(generated.strip()) > 0
        return score, is_greedy

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for request in requests:
            context, continuation = request.args if hasattr(request, "args") else request
            results.append(self._proxy_loglikelihood(context, continuation))
        return results

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        results: list[float] = []
        for request in requests:
            string = request.args[0]
            score, _ = self._proxy_loglikelihood("", string)
            results.append(score)
        return results

    def generate_until(self, requests: list[Any]) -> list[str]:
        outputs: list[str] = []

        for request in requests:
            context, gen_kwargs = request.args if hasattr(request, "args") else request
            gen_kwargs = dict(gen_kwargs or {})

            until = gen_kwargs.get("until")
            stop = until if isinstance(until, list) else ([until] if until else None)
            max_tokens = int(gen_kwargs.get("max_gen_toks", self._max_gen_toks))
            temperature = float(gen_kwargs.get("temperature", self._temperature))
            top_p = float(gen_kwargs.get("top_p", self._top_p))

            payload = self._build_payload(
                context,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            response = self._cached_generate(payload, cache_method="generate_until")
            outputs.append(self._extract_text(response))

        return outputs

    def ping_health(self) -> dict[str, Any]:
        return self._request(self.health_url, method="GET")

    def test_generation(self, prompt: str) -> str:
        payload = self._build_payload(
            prompt,
            max_tokens=32,
            temperature=0.2,
            top_p=0.95,
            stop=None,
        )
        response = self._cached_generate(payload, cache_method="__main__test_generation")
        return self._extract_text(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the local FastAPI generation server wrapper."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="FastAPI server base URL.",
    )
    parser.add_argument(
        "--prompt",
        default="Write one sentence about evaluation pipelines.",
        help="Prompt to use for the test generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lm = LocalFastAPIGenerateLM(base_url=args.base_url)

    print(f"Pinging health check at {lm.health_url}")
    health = lm.ping_health()
    print("Health response:")
    print(json.dumps(health, indent=2))

    print()
    print("Test generation:")
    print(lm.test_generation(args.prompt))


if __name__ == "__main__":
    main()
