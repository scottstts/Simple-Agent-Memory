from __future__ import annotations

from typing import Callable, Coroutine

OLLAMA_BASE_URL = "http://localhost:11434/api/chat"


def _build_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }


def create_ollama_client(model: str, base_url: str = OLLAMA_BASE_URL) -> Callable[[str], str]:
    import httpx

    def call(prompt: str) -> str:
        resp = httpx.post(base_url, json=_build_payload(model, prompt), timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    return call


def create_ollama_async_client(model: str, base_url: str = OLLAMA_BASE_URL) -> Callable[[str], Coroutine]:
    import httpx

    async def call(prompt: str) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(base_url, json=_build_payload(model, prompt), timeout=300)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    return call
