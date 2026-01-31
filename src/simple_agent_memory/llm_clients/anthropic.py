from __future__ import annotations

from typing import Callable, Coroutine


def create_anthropic_client(model: str, max_tokens: int = 4096, **kwargs) -> Callable[[str], str]:
    from anthropic import Anthropic
    client = Anthropic(**kwargs)

    def call(prompt: str) -> str:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    return call


def create_anthropic_async_client(model: str, max_tokens: int = 4096, **kwargs) -> Callable[[str], Coroutine]:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(**kwargs)

    async def call(prompt: str) -> str:
        resp = await client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    return call
