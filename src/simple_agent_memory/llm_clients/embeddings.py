from __future__ import annotations

import os
from typing import Callable, Coroutine

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
LMSTUDIO_EMBED_URL = "http://localhost:1234/v1/embeddings"


def create_openai_embedder(model: str, **kwargs) -> Callable[[str], list[float]]:
    from openai import OpenAI
    client = OpenAI(**kwargs)

    def embed(text: str) -> list[float]:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    return embed


def create_openai_async_embedder(model: str, **kwargs) -> Callable[[str], Coroutine]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(**kwargs)

    async def embed(text: str) -> list[float]:
        resp = await client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    return embed


def create_gemini_embedder(model: str, **kwargs) -> Callable[[str], list[float]]:
    import google.generativeai as genai
    if "api_key" in kwargs:
        genai.configure(api_key=kwargs.pop("api_key"))

    def embed(text: str) -> list[float]:
        resp = genai.embed_content(model=model, content=text, **kwargs)
        return resp["embedding"]

    return embed


def create_ollama_embedder(model: str, base_url: str = OLLAMA_EMBED_URL) -> Callable[[str], list[float]]:
    import httpx

    def embed(text: str) -> list[float]:
        resp = httpx.post(base_url, json={"model": model, "input": text}, timeout=300)
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    return embed


def create_ollama_async_embedder(model: str, base_url: str = OLLAMA_EMBED_URL) -> Callable[[str], Coroutine]:
    import httpx

    async def embed(text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(base_url, json={"model": model, "input": text}, timeout=300)
            resp.raise_for_status()
            return resp.json()["embeddings"][0]

    return embed


def create_lmstudio_embedder(model: str, base_url: str = LMSTUDIO_EMBED_URL) -> Callable[[str], list[float]]:
    import httpx

    def embed(text: str) -> list[float]:
        resp = httpx.post(base_url, json={"model": model, "input": text}, timeout=300)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    return embed


def create_lmstudio_async_embedder(model: str, base_url: str = LMSTUDIO_EMBED_URL) -> Callable[[str], Coroutine]:
    import httpx

    async def embed(text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(base_url, json={"model": model, "input": text}, timeout=300)
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    return embed


def create_openrouter_embedder(model: str, api_key: str | None = None) -> Callable[[str], list[float]]:
    import httpx
    key = api_key or os.environ["OPENROUTER_API_KEY"]

    def embed(text: str) -> list[float]:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            json={"model": model, "input": text},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    return embed


def create_openrouter_async_embedder(model: str, api_key: str | None = None) -> Callable[[str], Coroutine]:
    import httpx
    key = api_key or os.environ["OPENROUTER_API_KEY"]

    async def embed(text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/embeddings",
                json={"model": model, "input": text},
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    return embed
