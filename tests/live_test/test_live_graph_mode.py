import json
from pathlib import Path

import pytest

from simple_agent_memory import Memory
from simple_agent_memory.storage.sqlite_store import SQLiteStore


pytestmark = pytest.mark.live


def _load_messages(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for session in data["sessions"]:
        for msg in session["messages"]:
            lines.append(f'{msg["role"]}: {msg["content"]}')
    return lines


@pytest.mark.asyncio
async def test_live_graph_mode_internal_llm_conflict(openai_llm, openai_embedder, live_log_dir):
    data_path = Path(__file__).parent / "live_test_mock_data" / "relationship_graph_conversations.json"
    conversation = "\n".join(_load_messages(data_path))

    db_path = live_log_dir / "memory.db"
    async with Memory(
        "casey",
        openai_llm,
        embed=openai_embedder,
        db_path=db_path,
        mode="graph",
    ) as mem:
        await mem.memorize(conversation)
        query = "Where does Casey work?"
        result = await mem.retrieve(query, graph_level="graph_then_vector")

    (live_log_dir / "graph_mode_retrieve.txt").write_text(result, encoding="utf-8")
    (live_log_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "mode": "graph",
                "tool_mode": False,
                "query": query,
                "notes": "Graph mode set to graph_then_vector (triplets first, vector fallback).",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    store = SQLiteStore(db_path)
    triplets = await store.get_triplets("casey")
    await store.close()

    active_work = [
        t for t in triplets
        if t.predicate == "works_at" and t.subject.lower() in {"user", "casey", "i"}
    ]
    assert len(active_work) == 1
    assert "OpenAI" in result or "openai" in result


@pytest.mark.asyncio
async def test_live_graph_mode_tool_mode(openai_embedder, live_log_dir):
    db_path = live_log_dir / "memory.db"

    async with Memory(
        "casey",
        lambda _p: "",  # should not be used in tool_mode
        embed=openai_embedder,
        db_path=db_path,
        mode="graph",
        tool_mode=True,
    ) as mem:
        await mem.memorize(
            "I work at Acme Corp.",
            triplets=[{"subject": "User", "predicate": "works_at", "object": "Acme Corp"}],
        )
        query = "Where does the user work?"
        result = await mem.retrieve(query, entities=["User"], graph_level="graph_only")

    (live_log_dir / "graph_tool_mode_retrieve.txt").write_text(result, encoding="utf-8")
    (live_log_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "mode": "graph",
                "tool_mode": True,
                "query": query,
                "entities": ["User"],
                "notes": "Tool-mode graph retrieval set to graph_only.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert "Acme" in result
