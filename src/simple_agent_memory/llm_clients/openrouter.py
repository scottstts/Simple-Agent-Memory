from __future__ import annotations

import os
from typing import Callable, Coroutine

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def _build_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def create_openrouter_client(model: str, api_key: str | None = None) -> Callable[[str], str]:
    import httpx
    key = api_key or os.environ["OPENROUTER_API_KEY"]

    def call(prompt: str) -> str:
        resp = httpx.post(OPENROUTER_BASE_URL, json=_build_payload(model, prompt), headers=_headers(key), timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return call


def create_openrouter_async_client(model: str, api_key: str | None = None) -> Callable[[str], Coroutine]:
    import httpx
    key = api_key or os.environ["OPENROUTER_API_KEY"]

    async def call(prompt: str) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(OPENROUTER_BASE_URL, json=_build_payload(model, prompt), headers=_headers(key), timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    return call
