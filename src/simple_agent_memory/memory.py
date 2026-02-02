from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from .long_term.file_memory import FileMemory
from .long_term.graph_memory import GraphMemory
from .maintenance import MaintenanceRunner
from .retrieval import RetrievalPipeline
from .short_term import ShortTermMemory
from .storage.sqlite_store import DEFAULT_DB_DIR, SQLiteStore
from .types import EmbedCallable, LLMCallable
from .vector.numpy_store import NumpyVectorStore


def _vector_db_path(db_path: str | Path | None) -> str:
    if db_path is None:
        return str(DEFAULT_DB_DIR / "vectors.db")
    path = Path(db_path)
    suffix = path.suffix or ".db"
    stem = path.with_suffix("").name
    return str(path.with_name(f"{stem}_vectors{suffix}"))


class Memory:
    def __init__(
        self,
        user_id: str,
        llm: LLMCallable,
        embed: EmbedCallable | None = None,
        db_path: str | Path | None = None,
        mode: Literal["file", "graph", "both"] = "file",
        tool_mode: bool = False,
    ):
        self.user_id = user_id
        self._llm = llm
        self._embed = embed
        self._mode = mode
        self._tool_mode = tool_mode

        self._storage = SQLiteStore(db_path)

        self._vector: NumpyVectorStore | None = None
        if embed is not None:
            self._vector = NumpyVectorStore(_vector_db_path(db_path))
        elif mode in ("graph", "both"):
            raise ValueError("embed callable is required for graph/both mode")

        self._file_memory: FileMemory | None = None
        if mode in ("file", "both"):
            self._file_memory = FileMemory(
                user_id,
                self._storage,
                llm,
                vector_store=self._vector,
                embed=embed,
                tool_mode=tool_mode,
            )

        self._graph_memory: GraphMemory | None = None
        if mode in ("graph", "both") and self._vector and embed:
            self._graph_memory = GraphMemory(
                user_id,
                self._storage,
                self._vector,
                llm,
                embed,
                tool_mode=tool_mode,
            )

        self._retrieval = RetrievalPipeline(self._storage, self._vector, llm, embed)
        self._maintenance = MaintenanceRunner(self._storage, llm, self._vector, embed)

    async def memorize(
        self,
        text: str,
        *,
        items: list[dict] | None = None,
        triplets: list[dict] | None = None,
    ) -> None:
        if self._file_memory:
            await self._file_memory.memorize(text, items=items)
        if self._graph_memory:
            await self._graph_memory.memorize(text, triplets=triplets)

    async def retrieve(
        self,
        query: str,
        max_tokens: int = 2000,
        *,
        search_query: str | None = None,
        entities: list[str] | None = None,
        file_level: str = "auto",
        graph_level: str = "graph_then_vector",
        file_categories: list[str] | None = None,
        use_pipeline: bool | None = None,
    ) -> str:
        if use_pipeline is None:
            use_pipeline = True
        if self._mode == "graph" and self._graph_memory:
            results = await self._graph_memory.retrieve(query, entities=entities, level=graph_level)
            graph_context = self._format_graph_results(results)
            if graph_context:
                return graph_context
            return ""

        file_context = ""
        if self._file_memory and not self._tool_mode:
            file_context = await self._file_memory.retrieve(
                query,
                level=file_level,
                categories=file_categories,
                search_query=search_query,
            )
        elif self._file_memory and self._tool_mode:
            file_context = await self._file_memory.retrieve(
                query,
                level=file_level,
                categories=file_categories,
                search_query=search_query,
            )

        graph_context = ""
        if self._mode == "both" and self._graph_memory:
            results = await self._graph_memory.retrieve(query, entities=entities, level=graph_level)
            graph_context = self._format_graph_results(results)

        pipeline_context = ""
        if use_pipeline and (self._tool_mode or self._vector or not file_context):
            pipeline_context = await self._retrieval.retrieve(
                query,
                self.user_id,
                max_tokens,
                search_query=search_query,
                use_llm_query=not self._tool_mode,
            )
        parts = [p for p in (file_context, graph_context, pipeline_context) if p]
        return "\n\n".join(parts)

    def checkpoint(self, thread_id: str) -> ShortTermMemory:
        return ShortTermMemory(thread_id, self._storage)

    async def maintain(self, schedule: Literal["nightly", "weekly", "monthly", "all"] = "all") -> dict:
        if not self._file_memory:
            return {}
        if schedule == "all":
            return await self._maintenance.run_all(self.user_id)
        runner = getattr(self._maintenance, schedule)
        return await runner(self.user_id)

    async def close(self) -> None:
        await self._storage.close()
        if self._vector:
            await self._vector.close()

    @staticmethod
    def _format_graph_results(results: list) -> str:
        if not results:
            return ""
        lines = ["=== RELEVANT MEMORIES ===\n"]
        for r in results[:10]:
            lines.append(f"[{r.timestamp.isoformat()}] (confidence: {r.score:.2f})")
            lines.append(f"{r.text}\n")
        lines.append("=== END MEMORIES ===")
        return "\n".join(lines)

    # ── Sync wrappers ──

    def memorize_sync(
        self,
        text: str,
        *,
        items: list[dict] | None = None,
        triplets: list[dict] | None = None,
    ) -> None:
        asyncio.run(self.memorize(text, items=items, triplets=triplets))

    def retrieve_sync(
        self,
        query: str,
        max_tokens: int = 2000,
        *,
        search_query: str | None = None,
        entities: list[str] | None = None,
        file_level: str = "auto",
        graph_level: str = "graph_then_vector",
        file_categories: list[str] | None = None,
        use_pipeline: bool | None = None,
    ) -> str:
        return asyncio.run(
            self.retrieve(
                query,
                max_tokens,
                search_query=search_query,
                entities=entities,
                file_level=file_level,
                graph_level=graph_level,
                file_categories=file_categories,
                use_pipeline=use_pipeline,
            )
        )

    def maintain_sync(self, schedule: Literal["nightly", "weekly", "monthly", "all"] = "all") -> dict:
        return asyncio.run(self.maintain(schedule))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
