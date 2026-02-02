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
async def test_live_file_mode_internal_llm(openai_llm, openai_embedder, live_log_dir):
    data_path = Path(__file__).parent / "live_test_mock_data" / "personal_assistant_conversations.json"
    conversation = "\n".join(_load_messages(data_path))

    db_path = live_log_dir / "memory.db"
    async with Memory(
        "jordan",
        openai_llm,
        embed=openai_embedder,
        db_path=db_path,
        mode="file",
    ) as mem:
        await mem.memorize(conversation)
        await mem.maintain("nightly")
        query = "What programming language does Jordan prefer?"
        result = await mem.retrieve(query, file_level="summaries")

    (live_log_dir / "file_mode_retrieve.txt").write_text(result, encoding="utf-8")
    (live_log_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "mode": "file",
                "tool_mode": False,
                "query": query,
                "notes": "Internal LLM extraction/classification; nightly maintenance produces summaries.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    store = SQLiteStore(db_path)
    categories = await store.list_categories("jordan")
    await store.close()

    assert len(categories) >= 1
    expected_markers = [
        "python",
        "acme",
        "shellfish",
        "milo",
        "obsidian",
        "q2",
        "forecast",
    ]
    lowered = result.lower()
    assert any(marker in lowered for marker in expected_markers)


@pytest.mark.asyncio
async def test_live_file_mode_tool_mode(openai_embedder, live_log_dir):
    db_path = live_log_dir / "memory.db"

    async with Memory(
        "jordan",
        lambda _p: "",  # should not be used in tool_mode
        embed=openai_embedder,
        db_path=db_path,
        mode="file",
        tool_mode=True,
    ) as mem:
        await mem.memorize(
            "I prefer Python for scripting and I work at Acme Corp.",
            items=[
                {"content": "User prefers Python for scripting", "category": "preferences"},
                {"content": "User works at Acme Corp", "category": "work"},
            ],
        )

        query = "What language does the user prefer?"
        result = await mem.retrieve(query, search_query="python", file_level="items")

    (live_log_dir / "tool_mode_retrieve.txt").write_text(result, encoding="utf-8")
    (live_log_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "mode": "file",
                "tool_mode": True,
                "query": query,
                "search_query": "python",
                "notes": "Tool-mode retrieval is query-driven; file-level set to items.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert "Python" in result or "python" in result


@pytest.mark.asyncio
async def test_live_both_mode_and_maintenance(openai_llm, openai_embedder, live_log_dir):
    data_path = Path(__file__).parent / "live_test_mock_data" / "personal_assistant_conversations.json"
    conversation = "\n".join(_load_messages(data_path))

    db_path = live_log_dir / "memory.db"
    async with Memory(
        "jordan",
        openai_llm,
        embed=openai_embedder,
        db_path=db_path,
        mode="both",
    ) as mem:
        await mem.memorize(conversation)
        stats = await mem.maintain("all")
        query = "Where does Jordan work?"
        result = await mem.retrieve(query, file_level="summaries", graph_level="graph_then_vector")

    (live_log_dir / "both_mode_retrieve.txt").write_text(result, encoding="utf-8")
    (live_log_dir / "maintenance_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    (live_log_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "mode": "both",
                "tool_mode": False,
                "query": query,
                "notes": "Both-mode retrieval is query-specific; file summaries after maintenance + graph_then_vector.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert "RELEVANT MEMORIES" in result or "##" in result
    assert "nightly" in stats and "weekly" in stats and "monthly" in stats
