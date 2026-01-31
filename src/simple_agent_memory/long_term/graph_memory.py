from __future__ import annotations

import inspect
from datetime import datetime, timezone

from ..llm import parse_bool_response, parse_json_response
from ..prompts import DETECT_CONFLICT, EXTRACT_TRIPLETS, GRAPH_PREDICATE_FILTER
from ..storage.base import Storage
from ..tool_instructions import GRAPH_MEMORY_TOOL_INSTRUCTIONS
from ..types import EmbedCallable, LLMCallable, RetrievalResult, Triplet, _now
from ..vector.base import VectorStore


class GraphMemory:
    tool_use_instruction = GRAPH_MEMORY_TOOL_INSTRUCTIONS

    def __init__(
        self,
        user_id: str,
        storage: Storage,
        vector_store: VectorStore,
        llm: LLMCallable | None = None,
        embed: EmbedCallable | None = None,
        tool_mode: bool = False,
    ):
        self.user_id = user_id
        self._storage = storage
        self._vector = vector_store
        if llm is None and not tool_mode:
            raise ValueError("llm is required unless tool_mode=True")
        if embed is None:
            raise ValueError("embed callable is required for graph memory")
        self._llm = llm
        self._embed = embed
        self._tool_mode = tool_mode

    async def memorize(self, text: str, *, triplets: list[dict] | None = None) -> list[Triplet]:
        resource_id = await self._storage.save_resource(self.user_id, text)
        if self._tool_mode:
            raw_triplets = triplets or []
        else:
            raw_triplets = await self._extract_triplets(text)
        saved: list[Triplet] = []

        for raw in raw_triplets:
            status = str(raw.get("status", "current")).lower()
            if status not in {"current", "past", "uncertain"}:
                status = "uncertain"
            triplet = Triplet(
                subject=raw["subject"], predicate=raw["predicate"],
                object=raw["object"], timestamp=_now(),
                active=True,
                status=status,
            )

            if not self._tool_mode and triplet.status == "current":
                existing = await self._storage.get_triplets(self.user_id, subject=triplet.subject)
                if existing and await self._has_conflict(triplet, existing):
                    await self._storage.deactivate_triplet(
                        self.user_id, triplet.subject, triplet.predicate
                    )

            await self._storage.save_triplet(self.user_id, triplet)
            saved.append(triplet)

        embedding = await self._get_embedding(text)
        await self._vector.add(
            id=resource_id, text=text, embedding=embedding,
            metadata={
                "user_id": self.user_id,
                "type": "conversation",
                "created_at": _now().isoformat(),
                "accessed_at": _now().isoformat(),
            },
        )
        return saved

    async def retrieve(
        self,
        query: str,
        *,
        entities: list[str] | None = None,
        level: str = "graph_then_vector",
    ) -> list[RetrievalResult]:
        graph_results: list[RetrievalResult] = []
        vector_results: list[RetrievalResult] = []

        if level in ("graph_only", "graph_then_vector"):
            if self._tool_mode:
                if entities:
                    graph_results = await self._graph_search_entities(entities)
            else:
                graph_results = await self._graph_search(query)
                if not graph_results:
                    graph_results = await self._graph_search_entities(["User"])

        if level in ("vector_only", "graph_then_vector"):
            vector_results = await self._vector_search(query)

        if level == "graph_only":
            return self._rank_graph_results(graph_results)
        if level == "vector_only":
            return vector_results
        return self._merge_results(self._rank_graph_results(graph_results), vector_results, prefer_order=True)

    async def _extract_triplets(self, text: str) -> list[dict]:
        return await parse_json_response(self._llm, EXTRACT_TRIPLETS.format(text=text))

    async def _has_conflict(self, new: Triplet, existing: list[Triplet]) -> bool:
        same_predicate = [t for t in existing if t.predicate == new.predicate]
        if not same_predicate:
            return False

        facts = "\n".join(f"- {t.subject} {t.predicate} {t.object}" for t in same_predicate)
        return await parse_bool_response(
            self._llm,
            DETECT_CONFLICT.format(
                subject=new.subject, predicate=new.predicate,
                new_value=new.object, existing_facts=facts,
            ),
        )

    async def _vector_search(self, query: str) -> list[RetrievalResult]:
        embedding = await self._get_embedding(query)
        results = await self._vector.search(
            embedding, top_k=20, filter={"user_id": self.user_id}
        )
        return [
            RetrievalResult(
                text=meta.get("text", rid),
                score=score,
                timestamp=self._parse_ts(meta.get("created_at")) if meta.get("created_at") else _now(),
                source="vector",
                created_at=self._parse_ts(meta.get("created_at")) if meta.get("created_at") else None,
                accessed_at=self._parse_ts(meta.get("accessed_at")) if meta.get("accessed_at") else None,
            )
            for rid, score, meta in results
        ]

    async def _graph_search(self, query: str) -> list[RetrievalResult]:
        entities = await self._extract_entities(query)
        return await self._graph_search_entities(entities, query=query)

    async def _graph_search_entities(self, entities: list[str], query: str | None = None) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        predicate_filter: set[str] | None = None

        if query:
            all_triplets: list[Triplet] = []
            for entity in entities:
                all_triplets.extend(await self._storage.get_triplets(self.user_id, subject=entity))
            predicates = sorted({t.predicate for t in all_triplets})
            if predicates:
                pred_text = "\n".join(f"- {p}" for p in predicates)
                try:
                    keep = await parse_json_response(
                        self._llm,
                        GRAPH_PREDICATE_FILTER.format(query=query, predicates=pred_text),
                    )
                    predicate_filter = {p for p in keep if p in predicates}
                except Exception:
                    predicate_filter = None
        for entity in entities:
            triplets = await self._storage.get_triplets(self.user_id, subject=entity)
            for t in triplets:
                if predicate_filter and t.predicate not in predicate_filter:
                    continue
                results.append(self._graph_result(t, 0.8))

                connected = await self._storage.get_triplets(self.user_id, subject=t.object)
                for ct in connected:
                    if predicate_filter and ct.predicate not in predicate_filter:
                        continue
                    results.append(self._graph_result(ct, 0.6))

        return results

    async def _extract_entities(self, query: str) -> list[str]:
        prompt = f"Extract entity names from this query. Return a JSON array of strings.\n\nQuery: {query}\n\nReturn ONLY valid JSON."
        return await parse_json_response(self._llm, prompt)

    async def _get_embedding(self, text: str) -> list[float]:
        result = self._embed(text)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        dt = datetime.fromisoformat(value)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    @staticmethod
    def _merge_results(*result_lists: list[RetrievalResult], prefer_order: bool = False) -> list[RetrievalResult]:
        seen: set[str] = set()
        merged: list[RetrievalResult] = []
        for results in result_lists:
            for r in results:
                if r.text not in seen:
                    seen.add(r.text)
                    merged.append(r)
        if not prefer_order:
            merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:20]

    @staticmethod
    def _graph_result(triplet: Triplet, score: float) -> RetrievalResult:
        status = triplet.status or "current"
        label = f"{triplet.subject} {triplet.predicate} {triplet.object} ({status})"
        return RetrievalResult(
            text=label,
            score=score,
            timestamp=triplet.timestamp,
            source="graph",
            created_at=triplet.timestamp,
        )

    @staticmethod
    def _rank_graph_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
        def status_rank(text: str) -> int:
            if text.endswith("(current)"):
                return 0
            if text.endswith("(uncertain)"):
                return 1
            if text.endswith("(past)") or text.endswith("(past_replaced)"):
                return 2
            return 3

        return sorted(results, key=lambda r: (status_rank(r.text), -r.score))
