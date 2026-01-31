from __future__ import annotations

import inspect

from ..llm import invoke, parse_bool_response, parse_json_response
from ..prompts import CLASSIFY_ITEMS, EVOLVE_SUMMARY, EXTRACT_ITEMS, SELECT_CATEGORIES, SUFFICIENCY_CHECK
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
        summaries: dict[str, str] | None = None,
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

        by_category: dict[str, list[str]] = {}
        saved_items: list[MemoryItem] = []

        for item in classified:
            cat = item.get("category", "general")
            content = item.get("content", "")
            by_category.setdefault(cat, []).append(content)

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

        if self._tool_mode:
            if summaries:
                for category, summary in summaries.items():
                    await self._storage.save_category(self.user_id, category, summary)
        else:
            for category, contents in by_category.items():
                existing = await self._storage.load_category(self.user_id, category) or ""
                updated = await self._evolve_summary(category, existing, contents)
                await self._storage.save_category(self.user_id, category, updated)

        return saved_items

    async def retrieve(
        self,
        query: str,
        *,
        level: str = "auto",
        categories: list[str] | None = None,
        search_query: str | None = None,
    ) -> str:
        if self._tool_mode and level == "auto":
            level = "summaries"

        effective_query = search_query or query
        if self._tool_mode:
            if level == "items" or level == "resources":
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
                if level == "resources":
                    resources = await self._storage.search_resources(self.user_id, effective_query)
                    if resources:
                        result = (result + "\n\n") if result else ""
                        result += "## Raw Context\n" + "\n---\n".join(resources[:3])
                return result

            # summaries only
            cats = categories or await self._storage.list_categories(self.user_id)
            summaries = {}
            for cat in cats:
                s = await self._storage.load_category(self.user_id, cat)
                if s:
                    summaries[cat] = s
            return self._format_summaries(summaries) if summaries else ""

        all_categories = await self._storage.list_categories(self.user_id)
        if not all_categories:
            return ""

        relevant = categories or await self._select_categories(query, all_categories)
        if not relevant:
            relevant = all_categories
        summaries = {}
        for cat in relevant:
            s = await self._storage.load_category(self.user_id, cat)
            if s:
                summaries[cat] = s

        if not summaries:
            return ""

        if level == "summaries":
            return self._format_summaries(summaries)

        if level == "items":
            items = await self._storage.search_items(self.user_id, effective_query)
            if items:
                item_text = "\n".join(f"- {i.content}" for i in items)
                for item in items:
                    item.access_count += 1
                    item.accessed_at = _now()
                    await self._storage.update_item(item)
                return self._format_summaries(summaries) + f"\n\n## Detailed Items\n{item_text}"
            return self._format_summaries(summaries)

        if level == "resources":
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
                return (
                    self._format_summaries(summaries)
                    + item_section
                    + "\n\n## Raw Context\n"
                    + "\n---\n".join(resources[:3])
                )
            return self._format_summaries(summaries) + item_section

        if await self._is_sufficient(query, summaries):
            return self._format_summaries(summaries)

        items = await self._storage.search_items(self.user_id, effective_query)
        if items:
            item_text = "\n".join(f"- {i.content}" for i in items)
            for item in items:
                item.access_count += 1
                item.accessed_at = _now()
                await self._storage.update_item(item)
            return self._format_summaries(summaries) + f"\n\n## Detailed Items\n{item_text}"

        resources = await self._storage.search_resources(self.user_id, effective_query)
        if resources:
            return self._format_summaries(summaries) + "\n\n## Raw Context\n" + "\n---\n".join(resources[:3])

        return self._format_summaries(summaries)

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

    async def _evolve_summary(self, category: str, existing: str, new_items: list[str]) -> str:
        items_text = "\n".join(f"- {m}" for m in new_items)
        return await invoke(
            self._llm,
            EVOLVE_SUMMARY.format(
                category=category,
                existing=existing or "No existing summary.",
                new_items=items_text,
            ),
        )

    async def _select_categories(self, query: str, categories: list[str]) -> list[str]:
        cat_text = ", ".join(categories)
        result = await parse_json_response(
            self._llm, SELECT_CATEGORIES.format(query=query, categories=cat_text)
        )
        return [c for c in result if c in categories]

    async def _is_sufficient(self, query: str, summaries: dict[str, str]) -> bool:
        summary_text = "\n\n".join(f"### {k}\n{v}" for k, v in summaries.items())
        return await parse_bool_response(
            self._llm, SUFFICIENCY_CHECK.format(query=query, summaries=summary_text)
        )

    @staticmethod
    def _format_summaries(summaries: dict[str, str]) -> str:
        parts = [
            f"## {cat}\n{FileMemory._clean_summary(cat, text)}"
            for cat, text in summaries.items()
        ]
        return "\n\n".join(parts)

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
