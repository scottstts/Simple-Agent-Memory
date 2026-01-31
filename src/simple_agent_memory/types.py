from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Union
from uuid import uuid4

LLMCallable = Callable[[str], Union[str, Awaitable[str]]]
EmbedCallable = Callable[[str], Union[list[float], Awaitable[list[float]]]]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _id() -> str:
    return uuid4().hex


@dataclass
class MemoryItem:
    id: str = field(default_factory=_id)
    user_id: str = ""
    content: str = ""
    category: str = "general"
    source_id: str = ""
    embedding: list[float] | None = None
    access_count: int = 0
    created_at: datetime = field(default_factory=_now)
    accessed_at: datetime = field(default_factory=_now)
    archived: bool = False


@dataclass
class Triplet:
    subject: str = ""
    predicate: str = ""
    object: str = ""
    timestamp: datetime = field(default_factory=_now)
    active: bool = True
    status: str = "current"


@dataclass
class Checkpoint:
    thread_id: str = ""
    step_id: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_now)


@dataclass
class RetrievalResult:
    text: str = ""
    score: float = 0.0
    timestamp: datetime = field(default_factory=_now)
    source: str = "unknown"
    item_id: str | None = None
    created_at: datetime | None = None
    accessed_at: datetime | None = None
