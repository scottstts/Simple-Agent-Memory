from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from ..types import Checkpoint, MemoryItem, Triplet, _id, _now

DEFAULT_DB_DIR = Path.home() / ".simple_agent_memory"

SCHEMA = """
CREATE TABLE IF NOT EXISTS resources (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    source_id TEXT,
    embedding TEXT,
    access_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    archived INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS categories (
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    summary TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, category)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    state TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (thread_id, step_id)
);
CREATE TABLE IF NOT EXISTS triplets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    active INTEGER DEFAULT 1,
    status TEXT DEFAULT 'current'
);
CREATE INDEX IF NOT EXISTS idx_items_user ON items(user_id);
CREATE INDEX IF NOT EXISTS idx_items_category ON items(user_id, category);
CREATE INDEX IF NOT EXISTS idx_triplets_user ON triplets(user_id);
CREATE INDEX IF NOT EXISTS idx_triplets_subject ON triplets(user_id, subject);
CREATE INDEX IF NOT EXISTS idx_resources_user ON resources(user_id);
"""


def _ts(dt: datetime) -> str:
    return dt.isoformat()


def _parse_ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _row_to_item(row: aiosqlite.Row) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        user_id=row["user_id"],
        content=row["content"],
        category=row["category"],
        source_id=row["source_id"] or "",
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        access_count=row["access_count"],
        created_at=_parse_ts(row["created_at"]),
        accessed_at=_parse_ts(row["accessed_at"]),
        archived=bool(row["archived"]),
    )


def _row_to_triplet(row: aiosqlite.Row) -> Triplet:
    return Triplet(
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        timestamp=_parse_ts(row["timestamp"]),
        active=bool(row["active"]),
        status=row["status"] if "status" in row.keys() else "current",
    )


class SQLiteStore:
    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = DEFAULT_DB_DIR / "memory.db"
        self._db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.executescript(SCHEMA)
            await self._ensure_triplet_status(self._db)
            await self._db.commit()
        return self._db

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── Resources ──

    async def save_resource(self, user_id: str, content: str) -> str:
        db = await self._conn()
        rid = _id()
        await db.execute(
            "INSERT INTO resources (id, user_id, content, created_at) VALUES (?, ?, ?, ?)",
            (rid, user_id, content, _ts(_now())),
        )
        await db.commit()
        return rid

    async def get_resource(self, resource_id: str) -> str | None:
        db = await self._conn()
        cur = await db.execute("SELECT content FROM resources WHERE id = ?", (resource_id,))
        row = await cur.fetchone()
        return row["content"] if row else None

    async def search_resources(self, user_id: str, query: str) -> list[str]:
        db = await self._conn()
        cur = await db.execute(
            "SELECT content FROM resources WHERE user_id = ? AND content LIKE ? ORDER BY created_at DESC LIMIT 20",
            (user_id, f"%{query}%"),
        )
        return [row["content"] async for row in cur]

    # ── Items ──

    async def save_item(self, item: MemoryItem) -> None:
        db = await self._conn()
        await db.execute(
            "INSERT OR REPLACE INTO items (id, user_id, content, category, source_id, embedding, access_count, created_at, accessed_at, archived) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.id, item.user_id, item.content, item.category, item.source_id,
                json.dumps(item.embedding) if item.embedding else None,
                item.access_count, _ts(item.created_at), _ts(item.accessed_at), int(item.archived),
            ),
        )
        await db.commit()

    async def get_items(self, user_id: str, category: str | None = None, limit: int = 100) -> list[MemoryItem]:
        db = await self._conn()
        if category:
            cur = await db.execute(
                "SELECT * FROM items WHERE user_id = ? AND category = ? AND archived = 0 ORDER BY created_at DESC LIMIT ?",
                (user_id, category, limit),
            )
        else:
            cur = await db.execute(
                "SELECT * FROM items WHERE user_id = ? AND archived = 0 ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
        return [_row_to_item(row) async for row in cur]

    async def get_item_by_id(self, item_id: str) -> MemoryItem | None:
        db = await self._conn()
        cur = await db.execute(
            "SELECT * FROM items WHERE id = ? AND archived = 0",
            (item_id,),
        )
        row = await cur.fetchone()
        return _row_to_item(row) if row else None

    async def search_items(self, user_id: str, query: str) -> list[MemoryItem]:
        db = await self._conn()
        cur = await db.execute(
            "SELECT * FROM items WHERE user_id = ? AND archived = 0 AND content LIKE ? ORDER BY created_at DESC LIMIT 50",
            (user_id, f"%{query}%"),
        )
        return [_row_to_item(row) async for row in cur]

    async def update_item(self, item: MemoryItem) -> None:
        db = await self._conn()
        await db.execute(
            "UPDATE items SET content=?, category=?, embedding=?, access_count=?, accessed_at=?, archived=? WHERE id=?",
            (
                item.content, item.category,
                json.dumps(item.embedding) if item.embedding else None,
                item.access_count, _ts(item.accessed_at), int(item.archived), item.id,
            ),
        )
        await db.commit()

    async def delete_items(self, item_ids: list[str]) -> None:
        if not item_ids:
            return
        db = await self._conn()
        placeholders = ",".join("?" * len(item_ids))
        await db.execute(f"DELETE FROM items WHERE id IN ({placeholders})", item_ids)
        await db.commit()

    async def get_items_older_than(self, user_id: str, days: int) -> list[MemoryItem]:
        db = await self._conn()
        cutoff = _ts(_now().replace(day=_now().day))  # simplified
        from datetime import timedelta
        cutoff = _ts(_now() - timedelta(days=days))
        cur = await db.execute(
            "SELECT * FROM items WHERE user_id = ? AND archived = 0 AND created_at < ? ORDER BY created_at",
            (user_id, cutoff),
        )
        return [_row_to_item(row) async for row in cur]

    async def get_items_not_accessed_since(self, user_id: str, days: int) -> list[MemoryItem]:
        from datetime import timedelta
        cutoff = _ts(_now() - timedelta(days=days))
        db = await self._conn()
        cur = await db.execute(
            "SELECT * FROM items WHERE user_id = ? AND archived = 0 AND accessed_at < ?",
            (user_id, cutoff),
        )
        return [_row_to_item(row) async for row in cur]

    async def get_high_access_items(self, user_id: str, min_count: int = 5) -> list[MemoryItem]:
        db = await self._conn()
        cur = await db.execute(
            "SELECT * FROM items WHERE user_id = ? AND archived = 0 AND access_count >= ? ORDER BY access_count DESC",
            (user_id, min_count),
        )
        return [_row_to_item(row) async for row in cur]

    async def get_all_items(self, user_id: str) -> list[MemoryItem]:
        db = await self._conn()
        cur = await db.execute(
            "SELECT * FROM items WHERE user_id = ? AND archived = 0", (user_id,)
        )
        return [_row_to_item(row) async for row in cur]

    # ── Categories ──

    async def save_category(self, user_id: str, category: str, summary: str) -> None:
        db = await self._conn()
        await db.execute(
            "INSERT OR REPLACE INTO categories (user_id, category, summary, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, category, summary, _ts(_now())),
        )
        await db.commit()

    async def load_category(self, user_id: str, category: str) -> str | None:
        db = await self._conn()
        cur = await db.execute(
            "SELECT summary FROM categories WHERE user_id = ? AND category = ?",
            (user_id, category),
        )
        row = await cur.fetchone()
        return row["summary"] if row else None

    async def list_categories(self, user_id: str) -> list[str]:
        db = await self._conn()
        cur = await db.execute(
            "SELECT category FROM categories WHERE user_id = ? ORDER BY category", (user_id,)
        )
        return [row["category"] async for row in cur]

    # ── Checkpoints ──

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        db = await self._conn()
        await db.execute(
            "INSERT OR REPLACE INTO checkpoints (thread_id, step_id, state, timestamp) VALUES (?, ?, ?, ?)",
            (checkpoint.thread_id, checkpoint.step_id, json.dumps(checkpoint.state), _ts(checkpoint.timestamp)),
        )
        await db.commit()

    async def get_latest_checkpoint(self, thread_id: str) -> Checkpoint | None:
        db = await self._conn()
        cur = await db.execute(
            "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY timestamp DESC LIMIT 1",
            (thread_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return Checkpoint(
            thread_id=row["thread_id"], step_id=row["step_id"],
            state=json.loads(row["state"]), timestamp=_parse_ts(row["timestamp"]),
        )

    async def get_checkpoint_at_step(self, thread_id: str, step_id: str) -> Checkpoint | None:
        db = await self._conn()
        cur = await db.execute(
            "SELECT * FROM checkpoints WHERE thread_id = ? AND step_id = ?",
            (thread_id, step_id),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return Checkpoint(
            thread_id=row["thread_id"], step_id=row["step_id"],
            state=json.loads(row["state"]), timestamp=_parse_ts(row["timestamp"]),
        )

    async def list_checkpoint_steps(self, thread_id: str) -> list[str]:
        db = await self._conn()
        cur = await db.execute(
            "SELECT step_id FROM checkpoints WHERE thread_id = ? ORDER BY timestamp", (thread_id,)
        )
        return [row["step_id"] async for row in cur]

    # ── Triplets ──

    async def save_triplet(self, user_id: str, triplet: Triplet) -> None:
        db = await self._conn()
        await db.execute(
            "INSERT INTO triplets (user_id, subject, predicate, object, timestamp, active, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                triplet.subject,
                triplet.predicate,
                triplet.object,
                _ts(triplet.timestamp),
                int(triplet.active),
                triplet.status,
            ),
        )
        await db.commit()

    async def get_triplets(self, user_id: str, subject: str | None = None) -> list[Triplet]:
        db = await self._conn()
        if subject:
            cur = await db.execute(
                "SELECT * FROM triplets WHERE user_id = ? AND subject = ? AND active = 1",
                (user_id, subject),
            )
        else:
            cur = await db.execute(
                "SELECT * FROM triplets WHERE user_id = ? AND active = 1", (user_id,)
            )
        return [_row_to_triplet(row) async for row in cur]

    async def deactivate_triplet(self, user_id: str, subject: str, predicate: str) -> None:
        db = await self._conn()
        await db.execute(
            "UPDATE triplets SET active = 0, status = 'past_replaced' WHERE user_id = ? AND subject = ? AND predicate = ? AND active = 1",
            (user_id, subject, predicate),
        )
        await db.commit()

    async def _ensure_triplet_status(self, db: aiosqlite.Connection) -> None:
        cur = await db.execute("PRAGMA table_info(triplets)")
        cols = [row["name"] async for row in cur]
        if "status" not in cols:
            await db.execute("ALTER TABLE triplets ADD COLUMN status TEXT DEFAULT 'current'")

    async def get_all_triplets(self, user_id: str) -> list[Triplet]:
        db = await self._conn()
        cur = await db.execute(
            "SELECT * FROM triplets WHERE user_id = ? AND active = 1", (user_id,)
        )
        return [_row_to_triplet(row) async for row in cur]
