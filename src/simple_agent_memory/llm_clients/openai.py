from __future__ import annotations

from typing import Callable, Coroutine


def create_openai_client(model: str, **kwargs) -> Callable[[str], str]:
    from openai import OpenAI
    client = OpenAI(**kwargs)

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content or ""

    return call


def create_openai_async_client(model: str, **kwargs) -> Callable[[str], Coroutine]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(**kwargs)

    async def call(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content or ""

    return call
