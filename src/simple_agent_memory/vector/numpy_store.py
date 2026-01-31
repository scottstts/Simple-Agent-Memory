from __future__ import annotations

import json
from typing import Any

import aiosqlite
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    metadata TEXT
);
"""


class NumpyVectorStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] = []
        self._loaded = False

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
            await self._db.executescript(SCHEMA)
            await self._db.commit()
        return self._db

    async def _load(self) -> None:
        if self._loaded:
            return
        db = await self._conn()
        cur = await db.execute("SELECT id, text, embedding, metadata FROM vectors")
        rows = await cur.fetchall()
        self._ids = [r["id"] for r in rows]
        self._texts = [r["text"] for r in rows]
        self._embeddings = (
            np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            if rows else None
        )
        self._metadata = [json.loads(r["metadata"]) if r["metadata"] else {} for r in rows]
        self._loaded = True

    async def add(self, id: str, text: str, embedding: list[float], metadata: dict[str, Any] | None = None) -> None:
        db = await self._conn()
        emb = np.array(embedding, dtype=np.float32)
        meta = json.dumps(metadata) if metadata else None
        await db.execute(
            "INSERT OR REPLACE INTO vectors (id, text, embedding, metadata) VALUES (?, ?, ?, ?)",
            (id, text, emb.tobytes(), meta),
        )
        await db.commit()

        # Update in-memory index
        if id in self._ids:
            idx = self._ids.index(id)
            self._texts[idx] = text
            self._metadata[idx] = metadata or {}
            if self._embeddings is not None:
                self._embeddings[idx] = emb
        else:
            self._ids.append(id)
            self._texts.append(text)
            self._metadata.append(metadata or {})
            if self._embeddings is not None:
                self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])
            else:
                self._embeddings = emb.reshape(1, -1)

    async def search(self, embedding: list[float], top_k: int = 20, filter: dict[str, Any] | None = None) -> list[tuple[str, float, dict[str, Any]]]:
        await self._load()
        if self._embeddings is None or len(self._ids) == 0:
            return []

        query = np.array(embedding, dtype=np.float32)
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query)
        norms = np.where(norms == 0, 1e-10, norms)
        scores = self._embeddings @ query / norms

        if filter:
            mask = np.ones(len(self._ids), dtype=bool)
            for k, v in filter.items():
                mask &= np.array([m.get(k) == v for m in self._metadata])
            scores = np.where(mask, scores, -1.0)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[tuple[str, float, dict[str, Any]]] = []
        for i in top_indices:
            if scores[i] <= 0:
                continue
            meta = dict(self._metadata[i])
            meta.setdefault("text", self._texts[i])
            results.append((self._ids[i], float(scores[i]), meta))
        return results

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        db = await self._conn()
        placeholders = ",".join("?" * len(ids))
        await db.execute(f"DELETE FROM vectors WHERE id IN ({placeholders})", ids)
        await db.commit()
        self._loaded = False  # force reload

    async def rebuild(self) -> None:
        self._loaded = False
        await self._load()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
