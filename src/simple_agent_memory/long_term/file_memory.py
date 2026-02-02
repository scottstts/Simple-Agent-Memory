from __future__ import annotations

import inspect

from ..llm import parse_bool_response, parse_json_response
from ..prompts import CLASSIFY_ITEMS, EXTRACT_ITEMS, SELECT_CATEGORIES, SUFFICIENCY_CHECK
from ..storage.base import Storage
from ..types import EmbedCallable, LLMCallable, MemoryItem, _id, _now
from ..vector.base import VectorStore
from ..tool_instructions import FILE_MEMORY_TOOL_INSTRUCTIONS


class FileMemory:
    tool_use_instruction = FILE_MEMORY_TOOL_INSTRUCTIONS

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
        vector_only = level == "vector_only"
        vector_after = level.endswith("_then_vector")
        base_level = level[:-len("_then_vector")] if vector_after else level

        if self._tool_mode and base_level == "auto":
            base_level = "summaries"

        effective_query = search_query or query
        if vector_only:
            return await self._format_vector_results(effective_query)

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
                if vector_after:
                    vector_section = await self._format_vector_results(effective_query)
                    if vector_section:
                        return result + ("\n\n" if result else "") + vector_section
                return result

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
            if vector_after:
                vector_section = await self._format_vector_results(effective_query)
                if vector_section:
                    return result + ("\n\n" if result else "") + vector_section
            return result

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
            if vector_after:
                vector_section = await self._format_vector_results(effective_query)
                if vector_section:
                    return result + ("\n\n" if result else "") + vector_section
            return result

        if base_level == "items":
            items = await self._storage.search_items(self.user_id, effective_query)
            if items:
                item_text = "\n".join(f"- {i.content}" for i in items)
                for item in items:
                    item.access_count += 1
                    item.accessed_at = _now()
                    await self._storage.update_item(item)
                result = self._format_summaries(general, persistent) + f"\n\n## Detailed Items\n{item_text}"
                if vector_after:
                    vector_section = await self._format_vector_results(effective_query)
                    if vector_section:
                        return result + "\n\n" + vector_section
                return result
            result = self._format_summaries(general, persistent)
            if vector_after:
                vector_section = await self._format_vector_results(effective_query)
                if vector_section:
                    return result + ("\n\n" if result else "") + vector_section
            return result

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
                if vector_after:
                    vector_section = await self._format_vector_results(effective_query)
                    if vector_section:
                        return result + "\n\n" + vector_section
                return result
            result = self._format_summaries(general, persistent) + item_section
            if vector_after:
                vector_section = await self._format_vector_results(effective_query)
                if vector_section:
                    return result + ("\n\n" if result else "") + vector_section
            return result

        if general or persistent:
            if await self._is_sufficient(query, general, persistent):
                result = self._format_summaries(general, persistent)
                if vector_after:
                    vector_section = await self._format_vector_results(effective_query)
                    if vector_section:
                        return result + ("\n\n" if result else "") + vector_section
                return result

        items = await self._storage.search_items(self.user_id, effective_query)
        if items:
            item_text = "\n".join(f"- {i.content}" for i in items)
            for item in items:
                item.access_count += 1
                item.accessed_at = _now()
                await self._storage.update_item(item)
            result = self._format_summaries(general, persistent) + f"\n\n## Detailed Items\n{item_text}"
            if vector_after:
                vector_section = await self._format_vector_results(effective_query)
                if vector_section:
                    return result + "\n\n" + vector_section
            return result

        resources = await self._storage.search_resources(self.user_id, effective_query)
        if resources:
            result = (
                self._format_summaries(general, persistent)
                + "\n\n## Raw Context\n"
                + "\n---\n".join(resources[:3])
            )
            if vector_after:
                vector_section = await self._format_vector_results(effective_query)
                if vector_section:
                    return result + "\n\n" + vector_section
            return result

        result = self._format_summaries(general, persistent)
        if vector_after:
            vector_section = await self._format_vector_results(effective_query)
            if vector_section:
                return result + ("\n\n" if result else "") + vector_section
        return result

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

    async def _format_vector_results(self, query: str) -> str:
        results = await self._vector_search(query)
        if not results:
            return ""
        lines = ["## Semantic Items"]
        updated: set[str] = set()
        for text, score, item_id, category in results:
            cat_label = f" [{category}]" if category else ""
            lines.append(f"- {text}{cat_label} (score: {score:.2f})")
            if item_id and item_id not in updated:
                item = await self._storage.get_item_by_id(item_id)
                if item:
                    item.access_count += 1
                    item.accessed_at = _now()
                    await self._storage.update_item(item)
                    updated.add(item_id)
        return "\n".join(lines)

    async def _vector_search(self, query: str) -> list[tuple[str, float, str | None, str | None]]:
        if not self._vector or not self._embed:
            return []
        result = self._embed(query)
        embedding = await result if inspect.isawaitable(result) else result
        results = await self._vector.search(
            embedding, top_k=20, filter={"user_id": self.user_id, "type": "item"}
        )
        output: list[tuple[str, float, str | None, str | None]] = []
        for rid, score, meta in results:
            text = meta.get("text", rid)
            item_id = rid if meta.get("type") == "item" else None
            category = meta.get("category")
            output.append((text, score, item_id, category))
        return output

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
