from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from .llm import invoke
from .prompts import GENERATE_QUERY
from .storage.base import Storage
from .types import EmbedCallable, LLMCallable, MemoryItem, RetrievalResult, _now
from .vector.base import VectorStore


class RetrievalPipeline:
    def __init__(self, storage: Storage, vector_store: VectorStore | None,
                 llm: LLMCallable, embed: EmbedCallable | None,
                 relevance_threshold: float = 0.7, decay_half_life_days: float = 30.0,
                 access_weight: float = 0.7):
        self._storage = storage
        self._vector = vector_store
        self._llm = llm
        self._embed = embed
        self._threshold = relevance_threshold
        self._decay_hl = decay_half_life_days
        self._access_weight = min(max(access_weight, 0.0), 1.0)

    async def retrieve(
        self,
        user_message: str,
        user_id: str,
        max_tokens: int = 2000,
        *,
        search_query: str | None = None,
        use_llm_query: bool = True,
    ) -> str:
        if search_query is None:
            search_query = await self._generate_query(user_message) if use_llm_query else user_message

        candidates = await self._search(search_query, user_id)

        if not candidates:
            return ""

        item_map: dict[str, MemoryItem] = {
            item.id: item for _, item in candidates if item is not None
        }

        scored: list[tuple[float, RetrievalResult]] = []
        for result, _ in candidates:
            if result.score < self._threshold:
                continue
            scored.append((self._apply_decay(result), result))
        scored.sort(key=lambda x: x[0], reverse=True)

        selected: list[RetrievalResult] = []
        token_count = 0
        for score, mem in scored:
            tokens = len(mem.text) // 4
            if token_count + tokens > max_tokens:
                break
            mem.score = score
            selected.append(mem)
            token_count += tokens

        updated: set[str] = set()
        for mem in selected:
            if not mem.item_id or mem.item_id in updated:
                continue
            item = item_map.get(mem.item_id)
            if item is None:
                item = await self._storage.get_item_by_id(mem.item_id)
            if item:
                item.access_count += 1
                item.accessed_at = _now()
                await self._storage.update_item(item)
                updated.add(mem.item_id)

        return self._format_context(selected)

    async def _generate_query(self, message: str) -> str:
        return await invoke(self._llm, GENERATE_QUERY.format(message=message))

    async def _search(self, query: str, user_id: str) -> list[tuple[RetrievalResult, MemoryItem | None]]:
        results: list[tuple[RetrievalResult, MemoryItem | None]] = []

        items = await self._storage.search_items(user_id, query)
        for item in items:
            results.append((
                RetrievalResult(
                    text=item.content,
                    score=0.8,
                    timestamp=item.created_at,
                    source="file",
                    item_id=item.id,
                    created_at=item.created_at,
                    accessed_at=item.accessed_at,
                ),
                item,
            ))

        if self._vector and self._embed:
            emb_result = self._embed(query)
            if asyncio.iscoroutine(emb_result):
                embedding = await emb_result
            else:
                embedding = emb_result
            vec_results = await self._vector.search(embedding, top_k=20, filter={"user_id": user_id})
            for rid, score, meta in vec_results:
                created_at = self._parse_ts(meta.get("created_at")) if meta.get("created_at") else None
                accessed_at = self._parse_ts(meta.get("accessed_at")) if meta.get("accessed_at") else None
                results.append((
                    RetrievalResult(
                        text=meta.get("text", rid),
                        score=score,
                        timestamp=created_at or _now(),
                        source="vector",
                        item_id=rid if meta.get("type") == "item" else None,
                        created_at=created_at,
                        accessed_at=accessed_at,
                    ),
                    None,
                ))

        return results

    def _apply_decay(self, result: RetrievalResult) -> float:
        now = _now()
        created = result.created_at or result.timestamp
        accessed = result.accessed_at or result.timestamp
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if accessed.tzinfo is None:
            accessed = accessed.replace(tzinfo=timezone.utc)
        if accessed < created:
            accessed = created
        blended = created + (accessed - created) * self._access_weight
        age_days = (now - blended).total_seconds() / 86400
        return result.score / (1.0 + age_days / self._decay_hl)

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        dt = datetime.fromisoformat(value)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    @staticmethod
    def _format_context(memories: list[RetrievalResult]) -> str:
        if not memories:
            return ""
        lines = ["=== RELEVANT MEMORIES ===\n"]
        for m in memories:
            lines.append(f"[{m.timestamp.isoformat()}] (confidence: {m.score:.2f})")
            lines.append(f"{m.text}\n")
        lines.append("=== END MEMORIES ===")
        return "\n".join(lines)
