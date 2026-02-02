from __future__ import annotations

import inspect
from datetime import datetime, timezone

from ..llm import invoke, parse_bool_response, parse_json_response
from ..prompts import CLASSIFY_ITEMS, EXTRACT_ITEMS, GENERATE_QUERY, SELECT_CATEGORIES, SUFFICIENCY_CHECK
from ..storage.base import Storage
from ..types import EmbedCallable, LLMCallable, MemoryItem, RetrievalResult, _id, _now
from ..vector.base import VectorStore
from ..tool_instructions import FILE_MEMORY_TOOL_INSTRUCTIONS


class FileMemory:
    tool_use_instruction = FILE_MEMORY_TOOL_INSTRUCTIONS
    _RELEVANCE_THRESHOLD = 0.7
    _DECAY_HALF_LIFE_DAYS = 30.0
    _ACCESS_WEIGHT = 0.7

    def __init__(
        self,
        user_id: str,
        storage: Storage,
        llm: LLMCallable | None = None,
        vector_store: VectorStore | None = None,
        embed: EmbedCallable | None = None,
        tool_mode: bool = False,
    ):
        self.user_id = user_id
        self._storage = storage
        if llm is None and not tool_mode:
            raise ValueError("llm is required unless tool_mode=True")
        self._llm = llm
        self._vector = vector_store
        self._embed = embed
        self._tool_mode = tool_mode

    async def memorize(
        self,
        text: str,
        *,
        items: list[dict] | None = None,
    ) -> list[MemoryItem]:
        resource_id = await self._storage.save_resource(self.user_id, text)
        if self._tool_mode:
            raw_items = items if items is not None else ([{"content": text, "category": "general"}] if text else [])
            classified = []
            for item in raw_items:
                cat = item.get("category") or item.get("category_hint") or "general"
                classified.append({"content": item.get("content", ""), "category": cat})
        else:
            raw_items = await self._extract_items(text)
            classified = await self._classify_items(raw_items)

        saved_items: list[MemoryItem] = []

        for item in classified:
            cat = item.get("category", "general")
            content = item.get("content", "")

            mem = MemoryItem(
                id=_id(), user_id=self.user_id, content=content,
                category=cat, source_id=resource_id,
            )
            if self._embed and self._vector:
                result = self._embed(mem.content)
                mem.embedding = await result if inspect.isawaitable(result) else result
            await self._storage.save_item(mem)
            if self._vector and mem.embedding:
                await self._vector.add(
                    id=mem.id,
                    text=mem.content,
                    embedding=mem.embedding,
                    metadata={
                        "user_id": self.user_id,
                        "category": mem.category,
                        "type": "item",
                        "created_at": mem.created_at.isoformat(),
                        "accessed_at": mem.accessed_at.isoformat(),
                    },
                )
            saved_items.append(mem)

        return saved_items

    async def retrieve(
        self,
        query: str,
        *,
        level: str = "auto",
        categories: list[str] | None = None,
        search_query: str | None = None,
    ) -> str:
        level = self._normalize_level(level)
        semantic_only = level == "semantic"
        semantic_after = level.endswith("_then_semantic")
        base_level = level[:-len("_then_semantic")] if semantic_after else level

        if self._tool_mode and base_level == "auto":
            base_level = "summaries"

        if semantic_only:
            return await self._semantic_context(
                query,
                search_query=search_query,
                use_llm_query=not self._tool_mode,
            )

        semantic_context = ""
        if semantic_after:
            semantic_context = await self._semantic_context(
                query,
                search_query=search_query,
                use_llm_query=not self._tool_mode,
            )

        def append_semantic(result: str) -> str:
            if semantic_context:
                return result + ("\n\n" if result else "") + semantic_context
            return result

        effective_query = search_query or query
        if self._tool_mode:
            if base_level == "items" or base_level == "resources":
                items = await self._storage.search_items(self.user_id, effective_query)
                if items:
                    for item in items:
                        item.access_count += 1
                        item.accessed_at = _now()
                        await self._storage.update_item(item)
                    item_text = "\n".join(f"- {i.content}" for i in items)
                    result = f"## Retrieved Items\n{item_text}"
                else:
                    result = ""
                if base_level == "resources":
                    resources = await self._storage.search_resources(self.user_id, effective_query)
                    if resources:
                        result = (result + "\n\n") if result else ""
                        result += "## Raw Context\n" + "\n---\n".join(resources[:3])
                return append_semantic(result)

            # summaries only
            cats = categories or await self._storage.list_categories(self.user_id)
            general: dict[str, str] = {}
            persistent: dict[str, str] = {}
            for cat in cats:
                g = await self._storage.load_category(self.user_id, cat)
                if g:
                    general[cat] = g
                p = await self._storage.load_persistent_category(self.user_id, cat)
                if p:
                    persistent[cat] = p
            result = self._format_summaries(general, persistent) if general or persistent else ""
            return append_semantic(result)

        all_categories = await self._storage.list_categories(self.user_id)
        relevant = categories or (await self._select_categories(query, all_categories) if all_categories else [])
        if not relevant and all_categories:
            relevant = all_categories
        general: dict[str, str] = {}
        persistent: dict[str, str] = {}
        for cat in relevant:
            g = await self._storage.load_category(self.user_id, cat)
            if g:
                general[cat] = g
            p = await self._storage.load_persistent_category(self.user_id, cat)
            if p:
                persistent[cat] = p

        if base_level == "summaries":
            result = self._format_summaries(general, persistent)
            return append_semantic(result)

        if base_level == "items":
            items = await self._storage.search_items(self.user_id, effective_query)
            if items:
                item_text = "\n".join(f"- {i.content}" for i in items)
                for item in items:
                    item.access_count += 1
                    item.accessed_at = _now()
                    await self._storage.update_item(item)
                result = self._format_summaries(general, persistent) + f"\n\n## Detailed Items\n{item_text}"
                return append_semantic(result)
            result = self._format_summaries(general, persistent)
            return append_semantic(result)

        if base_level == "resources":
            items = await self._storage.search_items(self.user_id, effective_query)
            item_section = ""
            if items:
                item_text = "\n".join(f"- {i.content}" for i in items)
                for item in items:
                    item.access_count += 1
                    item.accessed_at = _now()
                    await self._storage.update_item(item)
                item_section = f"\n\n## Detailed Items\n{item_text}"

            resources = await self._storage.search_resources(self.user_id, effective_query)
            if resources:
                result = (
                    self._format_summaries(general, persistent)
                    + item_section
                    + "\n\n## Raw Context\n"
                    + "\n---\n".join(resources[:3])
                )
                return append_semantic(result)
            result = self._format_summaries(general, persistent) + item_section
            return append_semantic(result)

        if general or persistent:
            if await self._is_sufficient(query, general, persistent):
                result = self._format_summaries(general, persistent)
                return append_semantic(result)

        items = await self._storage.search_items(self.user_id, effective_query)
        if items:
            item_text = "\n".join(f"- {i.content}" for i in items)
            for item in items:
                item.access_count += 1
                item.accessed_at = _now()
                await self._storage.update_item(item)
            result = self._format_summaries(general, persistent) + f"\n\n## Detailed Items\n{item_text}"
            return append_semantic(result)

        resources = await self._storage.search_resources(self.user_id, effective_query)
        if resources:
            result = (
                self._format_summaries(general, persistent)
                + "\n\n## Raw Context\n"
                + "\n---\n".join(resources[:3])
            )
            return append_semantic(result)

        result = self._format_summaries(general, persistent)
        return append_semantic(result)

    async def _extract_items(self, text: str) -> list[dict]:
        return await parse_json_response(self._llm, EXTRACT_ITEMS.format(text=text))

    async def _classify_items(self, items: list[dict]) -> list[dict]:
        categories = await self._storage.list_categories(self.user_id)
        if not categories and all("category_hint" in i for i in items):
            for i in items:
                i["category"] = i.get("category_hint", "general")
            return items

        items_text = "\n".join(f'- {i.get("content", "")}' for i in items)
        cat_text = ", ".join(categories) if categories else "(none yet — create new ones)"
        result = await parse_json_response(
            self._llm, CLASSIFY_ITEMS.format(categories=cat_text, items=items_text)
        )
        return result

    async def _select_categories(self, query: str, categories: list[str]) -> list[str]:
        cat_text = ", ".join(categories)
        result = await parse_json_response(
            self._llm, SELECT_CATEGORIES.format(query=query, categories=cat_text)
        )
        return [c for c in result if c in categories]

    async def _is_sufficient(self, query: str, general: dict[str, str], persistent: dict[str, str]) -> bool:
        parts: list[str] = []
        for cat in sorted(set(general) | set(persistent)):
            if cat in general:
                parts.append(f"### {cat} (general)\n{general[cat]}")
            if cat in persistent:
                parts.append(f"### {cat} (persistent)\n{persistent[cat]}")
        summary_text = "\n\n".join(parts)
        return await parse_bool_response(
            self._llm, SUFFICIENCY_CHECK.format(query=query, summaries=summary_text)
        )

    @staticmethod
    def _format_summaries(general: dict[str, str], persistent: dict[str, str]) -> str:
        parts: list[str] = []
        for cat in sorted(set(general) | set(persistent)):
            section_parts: list[str] = []
            if cat in general:
                section_parts.append(
                    "General summary (all items):\n" + FileMemory._clean_summary(cat, general[cat])
                )
            if cat in persistent:
                section_parts.append(
                    "Persistent summary (long-lived items):\n" + FileMemory._clean_summary(cat, persistent[cat])
                )
            if section_parts:
                parts.append(f"## {cat}\n" + "\n\n".join(section_parts))
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_level(level: str) -> str:
        if level == "vector_only":
            return "semantic"
        if level.endswith("_then_vector"):
            return level.replace("_then_vector", "_then_semantic")
        return level

    async def _semantic_context(
        self,
        user_message: str,
        *,
        search_query: str | None,
        use_llm_query: bool,
        max_tokens: int = 2000,
    ) -> str:
        if search_query is None:
            if use_llm_query:
                search_query = await invoke(self._llm, GENERATE_QUERY.format(message=user_message))
            else:
                search_query = user_message

        candidates = await self._semantic_candidates(search_query)
        if not candidates:
            return ""

        item_map: dict[str, MemoryItem] = {
            item.id: item for _, item in candidates if item is not None
        }

        scored: list[tuple[float, RetrievalResult]] = []
        for result, _ in candidates:
            if result.score < self._RELEVANCE_THRESHOLD:
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

    async def _semantic_candidates(
        self, query: str
    ) -> list[tuple[RetrievalResult, MemoryItem | None]]:
        results: list[tuple[RetrievalResult, MemoryItem | None]] = []

        items = await self._storage.search_items(self.user_id, query)
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
            embedding = await emb_result if inspect.isawaitable(emb_result) else emb_result
            vec_results = await self._vector.search(
                embedding, top_k=20, filter={"user_id": self.user_id, "type": "item"}
            )
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
        blended = created + (accessed - created) * self._ACCESS_WEIGHT
        age_days = (now - blended).total_seconds() / 86400
        return result.score / (1.0 + age_days / self._DECAY_HALF_LIFE_DAYS)

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

    @staticmethod
    def _clean_summary(category: str, text: str) -> str:
        lines = [l for l in text.strip().splitlines() if l.strip()]
        while lines:
            if lines[0].lstrip().startswith("#"):
                heading = lines[0].lstrip("#").strip().lower()
                if heading == category.lower():
                    lines.pop(0)
                    continue
            break
        return "\n".join(lines).strip()
