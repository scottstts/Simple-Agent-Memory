from __future__ import annotations

from typing import Callable, Coroutine


def create_gemini_client(model: str, **kwargs) -> Callable[[str], str]:
    import google.generativeai as genai
    if "api_key" in kwargs:
        genai.configure(api_key=kwargs.pop("api_key"))
    gm = genai.GenerativeModel(model, **kwargs)

    def call(prompt: str) -> str:
        resp = gm.generate_content(prompt)
        return resp.text

    return call


def create_gemini_async_client(model: str, **kwargs) -> Callable[[str], Coroutine]:
    import google.generativeai as genai
    if "api_key" in kwargs:
        genai.configure(api_key=kwargs.pop("api_key"))
    gm = genai.GenerativeModel(model, **kwargs)

    async def call(prompt: str) -> str:
        resp = await gm.generate_content_async(prompt)
        return resp.text

    return call
