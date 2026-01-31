from __future__ import annotations

import asyncio

import numpy as np

from .llm import invoke
from .prompts import COMPRESS_MEMORIES, EVOLVE_SUMMARY
from .storage.base import Storage
from .types import EmbedCallable, LLMCallable, _now
from .vector.base import VectorStore


class MaintenanceRunner:
    def __init__(self, storage: Storage, llm: LLMCallable,
                 vector_store: VectorStore | None = None,
                 embed: EmbedCallable | None = None):
        self._storage = storage
        self._llm = llm
        self._vector = vector_store
        self._embed = embed

    async def nightly(self, user_id: str) -> dict:
        stats = {"merged": 0, "promoted": 0}

        items = await self._storage.get_all_items(user_id)
        if len(items) < 2:
            return stats

        embeddings = [i for i in items if i.embedding]
        if len(embeddings) >= 2:
            vecs = np.array([e.embedding for e in embeddings])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            normalized = vecs / norms
            sim_matrix = normalized @ normalized.T

            merged_ids: set[str] = set()
            for i in range(len(embeddings)):
                if embeddings[i].id in merged_ids:
                    continue
                group = [embeddings[i]]
                for j in range(i + 1, len(embeddings)):
                    if embeddings[j].id not in merged_ids and sim_matrix[i, j] > 0.95:
                        group.append(embeddings[j])
                        merged_ids.add(embeddings[j].id)
                if len(group) > 1:
                    combined = " | ".join(g.content for g in group)
                    merged_content = await invoke(
                        self._llm, COMPRESS_MEMORIES.format(items=combined)
                    )
                    group[0].content = merged_content
                    if self._embed and self._vector:
                        result = self._embed(group[0].content)
                        if asyncio.iscoroutine(result):
                            group[0].embedding = await result
                        else:
                            group[0].embedding = result
                    await self._storage.update_item(group[0])
                    deleted_ids = [g.id for g in group[1:]]
                    await self._storage.delete_items(deleted_ids)
                    if self._vector and deleted_ids:
                        await self._vector.delete(deleted_ids)
                    if self._vector and group[0].embedding:
                        await self._vector.add(
                            id=group[0].id,
                            text=group[0].content,
                            embedding=group[0].embedding,
                            metadata={
                                "user_id": user_id,
                                "category": group[0].category,
                                "type": "item",
                                "created_at": group[0].created_at.isoformat(),
                                "accessed_at": group[0].accessed_at.isoformat(),
                            },
                        )
                    stats["merged"] += len(group) - 1

        hot = await self._storage.get_high_access_items(user_id)
        for item in hot:
            item.access_count += 1
            await self._storage.update_item(item)
            stats["promoted"] += 1

        return stats

    async def weekly(self, user_id: str) -> dict:
        stats = {"summarized": 0, "archived": 0}

        old_items = await self._storage.get_items_older_than(user_id, days=30)
        by_cat: dict[str, list[str]] = {}
        for item in old_items:
            by_cat.setdefault(item.category, []).append(item.content)

        for cat, contents in by_cat.items():
            existing = await self._storage.load_category(user_id, cat) or ""
            items_text = "\n".join(f"- {c}" for c in contents)
            updated = await invoke(
                self._llm,
                EVOLVE_SUMMARY.format(category=cat, existing=existing, new_items=items_text),
            )
            await self._storage.save_category(user_id, cat, updated)
            stats["summarized"] += 1

        stale = await self._storage.get_items_not_accessed_since(user_id, days=90)
        for item in stale:
            item.archived = True
            await self._storage.update_item(item)
            if self._vector:
                await self._vector.delete([item.id])
            stats["archived"] += 1

        return stats

    async def monthly(self, user_id: str) -> dict:
        stats = {"reindexed": 0, "dead_archived": 0}

        if self._embed and self._vector:
            items = await self._storage.get_all_items(user_id)
            for item in items:
                result = self._embed(item.content)
                if asyncio.iscoroutine(result):
                    embedding = await result
                else:
                    embedding = result
                item.embedding = embedding
                await self._storage.update_item(item)
                await self._vector.add(
                    id=item.id, text=item.content, embedding=embedding,
                    metadata={
                        "user_id": user_id,
                        "category": item.category,
                        "type": "item",
                        "created_at": item.created_at.isoformat(),
                        "accessed_at": item.accessed_at.isoformat(),
                    },
                )
                stats["reindexed"] += 1

            await self._vector.rebuild()

        stale_items = await self._storage.get_items_not_accessed_since(user_id, days=180)
        for item in stale_items:
            item.archived = True
            await self._storage.update_item(item)
            if self._vector:
                await self._vector.delete([item.id])
            stats["dead_archived"] += 1

        return stats

    async def run_all(self, user_id: str) -> dict:
        n = await self.nightly(user_id)
        w = await self.weekly(user_id)
        m = await self.monthly(user_id)
        return {"nightly": n, "weekly": w, "monthly": m}
