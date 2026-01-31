from __future__ import annotations

import json
import re
import inspect
from typing import Any

from .types import LLMCallable


async def invoke(llm: LLMCallable, prompt: str) -> str:
    result = llm(prompt)
    if inspect.isawaitable(result):
        return await result
    return result


async def parse_json_response(llm: LLMCallable, prompt: str, max_retries: int = 2) -> Any:
    raw = await invoke(llm, prompt)
    for attempt in range(max_retries + 1):
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            if attempt >= max_retries:
                raise
            retry_prompt = (
                "Your previous response was invalid JSON.\n"
                f"Error: {exc}\n"
                "Expected JSON per the original instruction. "
                "Please output ONLY valid JSON and nothing else.\n\n"
                f"Original instruction:\n{prompt}\n\n"
                f"Invalid response:\n{raw}\n"
            )
            raw = await invoke(llm, retry_prompt)


async def parse_bool_response(llm: LLMCallable, prompt: str) -> bool:
    raw = await invoke(llm, prompt)
    return "YES" in raw.upper()
