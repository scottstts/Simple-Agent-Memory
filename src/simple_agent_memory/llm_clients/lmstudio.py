from __future__ import annotations

from typing import Callable, Coroutine

LMSTUDIO_BASE_URL = "http://localhost:1234/v1/chat/completions"


def _build_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }


def create_lmstudio_client(model: str, base_url: str = LMSTUDIO_BASE_URL) -> Callable[[str], str]:
    import httpx

    def call(prompt: str) -> str:
        resp = httpx.post(base_url, json=_build_payload(model, prompt), timeout=300)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return call


def create_lmstudio_async_client(model: str, base_url: str = LMSTUDIO_BASE_URL) -> Callable[[str], Coroutine]:
    import httpx

    async def call(prompt: str) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(base_url, json=_build_payload(model, prompt), timeout=300)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    return call
