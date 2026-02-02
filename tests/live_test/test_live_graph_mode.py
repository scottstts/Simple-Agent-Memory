import json
from pathlib import Path

import pytest

from simple_agent_memory.long_term.graph_memory import GraphMemory
from simple_agent_memory.storage.sqlite_store import SQLiteStore
from simple_agent_memory.vector.numpy_store import NumpyVectorStore
from tests.live_test.llm_sim import graph_retrieval_plan, tool_mode_graph_triplets


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


def _format_graph_results(results) -> str:
    if not results:
        return ""
    lines = ["=== RELEVANT MEMORIES ===\n"]
    for r in results[:10]:
        lines.append(f"[{r.timestamp.isoformat()}] (confidence: {r.score:.2f})")
        lines.append(f"{r.text}\n")
    lines.append("=== END MEMORIES ===")
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_live_graph_mode_internal_llm_conflict(openai_llm, openai_embedder, live_log_dir):
    data_path = Path(__file__).parent / "live_test_mock_data" / "relationship_graph_conversations.json"
    data = _load_data(data_path)
    conversation = "\n".join(_load_messages(data))
    user_name = data.get("user_name", "User")

    db_path = live_log_dir / "memory.db"
    store = SQLiteStore(db_path)
    vector = NumpyVectorStore(str(_vector_db_path(db_path)))
    gm = GraphMemory(user_name.lower(), storage=store, vector_store=vector, llm=openai_llm, embed=openai_embedder)
    try:
        await gm.memorize(conversation)
        plan = graph_retrieval_plan(openai_llm, conversation, tool_mode=False)
        query = plan["query"]
        result = _format_graph_results(
            await gm.retrieve(
                query,
                level=plan.get("level", "graph_then_vector"),
                expand=plan.get("expand", "medium"),
            )
        )
    finally:
        await store.close()
        await vector.close()

    (live_log_dir / "graph_mode_retrieve.txt").write_text(result, encoding="utf-8")
    (live_log_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "mode": "graph",
                "tool_mode": False,
                "query": query,
                "retrieval_plan": plan,
                "notes": "Graph mode with LLM-generated retrieval plan.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_live_graph_mode_tool_mode(openai_llm, openai_embedder, live_log_dir):
    data_path = Path(__file__).parent / "live_test_mock_data" / "relationship_graph_conversations.json"
    data = _load_data(data_path)
    conversation = "\n".join(_load_messages(data))
    user_name = data.get("user_name", "User")

    db_path = live_log_dir / "memory.db"
    store = SQLiteStore(db_path)
    vector = NumpyVectorStore(str(_vector_db_path(db_path)))
    gm = GraphMemory(user_name.lower(), storage=store, vector_store=vector, llm=None, embed=openai_embedder, tool_mode=True)
    try:
        triplets = tool_mode_graph_triplets(openai_llm, conversation, user_name=user_name)
        await gm.memorize(conversation, triplets=triplets)
        plan = graph_retrieval_plan(openai_llm, conversation, tool_mode=True)
        query = plan["query"]
        result = _format_graph_results(
            await gm.retrieve(
                query,
                entities=plan.get("entities"),
                level=plan.get("level", "graph_only"),
                expand=plan.get("expand", "medium"),
            )
        )
    finally:
        await store.close()
        await vector.close()

    (live_log_dir / "graph_tool_mode_retrieve.txt").write_text(result, encoding="utf-8")
    (live_log_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "mode": "graph",
                "tool_mode": True,
                "query": query,
                "retrieval_plan": plan,
                "tool_inputs": {"triplets": triplets},
                "notes": "Tool-mode graph retrieval with LLM-generated triplets and plan.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
