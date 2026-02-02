import json
from pathlib import Path

import pytest

from simple_agent_memory.long_term.file_memory import FileMemory
from simple_agent_memory.maintenance import MaintenanceRunner
from simple_agent_memory.storage.sqlite_store import SQLiteStore
from simple_agent_memory.vector.numpy_store import NumpyVectorStore
from tests.live_test.llm_sim import (
    file_retrieval_plan,
    tool_mode_file_items,
)


pytestmark = pytest.mark.live


def _load_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_messages(data: dict) -> list[str]:
    lines: list[str] = []
    for session in data["sessions"]:
        for msg in session["messages"]:
            lines.append(f'{msg["role"]}: {msg["content"]}')
    return lines


def _vector_db_path(db_path: Path) -> Path:
    suffix = db_path.suffix or ".db"
    stem = db_path.with_suffix("").name
    return db_path.with_name(f"{stem}_vectors{suffix}")


@pytest.mark.asyncio
async def test_live_file_mode_internal_llm(openai_llm, openai_embedder, live_log_dir):
    data_path = Path(__file__).parent / "live_test_mock_data" / "personal_assistant_conversations.json"
    data = _load_data(data_path)
    conversation = "\n".join(_load_messages(data))
    user_name = data.get("user_name", "User")

    db_path = live_log_dir / "memory.db"
    store = SQLiteStore(db_path)
    vector = NumpyVectorStore(str(_vector_db_path(db_path)))
    fm = FileMemory(user_name.lower(), storage=store, llm=openai_llm, vector_store=vector, embed=openai_embedder)
    runner = MaintenanceRunner(store, openai_llm, vector, openai_embedder)
    try:
        await fm.memorize(conversation)
        await runner.nightly(user_name.lower())
        plan = file_retrieval_plan(openai_llm, conversation, tool_mode=False)
        query = plan["query"]
        result = await fm.retrieve(
            query,
            level=plan.get("level", "summaries"),
            categories=plan.get("categories"),
            search_query=plan.get("search_query"),
        )
    finally:
        await store.close()
        await vector.close()

    (live_log_dir / "file_mode_retrieve.txt").write_text(result, encoding="utf-8")
    (live_log_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "mode": "file",
                "tool_mode": False,
                "query": query,
                "retrieval_plan": plan,
                "notes": "Internal LLM extraction/classification; nightly maintenance produces summaries.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_live_file_mode_tool_mode(openai_llm, openai_embedder, live_log_dir):
    data_path = Path(__file__).parent / "live_test_mock_data" / "personal_assistant_conversations.json"
    data = _load_data(data_path)
    conversation = "\n".join(_load_messages(data))
    user_name = data.get("user_name", "User")

    db_path = live_log_dir / "memory.db"
    store = SQLiteStore(db_path)
    vector = NumpyVectorStore(str(_vector_db_path(db_path)))
    fm = FileMemory(
        user_name.lower(),
        storage=store,
        llm=None,
        vector_store=vector,
        embed=openai_embedder,
        tool_mode=True,
    )
    runner = MaintenanceRunner(store, openai_llm, vector, openai_embedder)
    try:
        items = tool_mode_file_items(openai_llm, conversation, user_name=user_name)
        await fm.memorize(conversation, items=items)
        await runner.nightly(user_name.lower())
        plan = file_retrieval_plan(openai_llm, conversation, tool_mode=True)
        query = plan["query"]
        result = await fm.retrieve(
            query,
            level=plan.get("level", "items"),
            categories=plan.get("categories"),
            search_query=plan.get("search_query"),
        )
    finally:
        await store.close()
        await vector.close()

    (live_log_dir / "tool_mode_retrieve.txt").write_text(result, encoding="utf-8")
    (live_log_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "mode": "file",
                "tool_mode": True,
                "query": query,
                "retrieval_plan": plan,
                "tool_inputs": {"items": items},
                "notes": "Tool-mode retrieval with LLM-generated items and query plan.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_live_maintenance(openai_llm, openai_embedder, live_log_dir):
    data_path = Path(__file__).parent / "live_test_mock_data" / "personal_assistant_conversations.json"
    data = _load_data(data_path)
    conversation = "\n".join(_load_messages(data))
    user_name = data.get("user_name", "User")

    db_path = live_log_dir / "memory.db"
    store = SQLiteStore(db_path)
    vector = NumpyVectorStore(str(_vector_db_path(db_path)))
    fm = FileMemory(user_name.lower(), storage=store, llm=openai_llm, vector_store=vector, embed=openai_embedder)
    runner = MaintenanceRunner(store, openai_llm, vector, openai_embedder)
    try:
        await fm.memorize(conversation)
        stats = await runner.run_all(user_name.lower())
    finally:
        await store.close()
        await vector.close()

    (live_log_dir / "maintenance_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    (live_log_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "mode": "maintenance",
                "tool_mode": False,
                "notes": "Maintenance-only run (nightly/weekly/monthly) after file memorize.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
