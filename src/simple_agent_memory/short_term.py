from __future__ import annotations

from typing import Any

from .storage.base import Storage
from .types import Checkpoint, _now


class ShortTermMemory:
    def __init__(self, thread_id: str, storage: Storage):
        self.thread_id = thread_id
        self._storage = storage

    async def load_latest(self) -> dict[str, Any] | None:
        cp = await self._storage.get_latest_checkpoint(self.thread_id)
        return cp.state if cp else None

    async def save(self, step_id: str, state: dict[str, Any]) -> Checkpoint:
        cp = Checkpoint(thread_id=self.thread_id, step_id=step_id, state=state, timestamp=_now())
        await self._storage.save_checkpoint(cp)
        return cp

    async def rewind(self, step_id: str) -> dict[str, Any] | None:
        cp = await self._storage.get_checkpoint_at_step(self.thread_id, step_id)
        return cp.state if cp else None

    async def list_steps(self) -> list[str]:
        return await self._storage.list_checkpoint_steps(self.thread_id)
