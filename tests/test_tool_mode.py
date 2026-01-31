import pytest

from simple_agent_memory import Memory
from simple_agent_memory.long_term.graph_memory import GraphMemory


def _raising_llm(_prompt: str) -> str:
    raise AssertionError("LLM should not be called in tool_mode")


@pytest.mark.asyncio
async def test_tool_mode_file_memory_no_llm(tmp_path, mock_embed):
    db = tmp_path / "mem.db"
    async with Memory(
        "u1",
        _raising_llm,
        embed=mock_embed,
        db_path=db,
        mode="file",
        tool_mode=True,
    ) as mem:
        await mem.memorize(
            "I prefer Python for scripting and I work at Acme.",
            items=[
                {"content": "User prefers Python for scripting", "category": "preferences"},
                {"content": "User works at Acme", "category": "work"},
            ],
            summaries={
                "preferences": "- Prefers Python for scripting",
                "work": "- Works at Acme",
            },
        )

        result = await mem.retrieve("irrelevant", search_query="Python")
        assert "Python" in result


@pytest.mark.asyncio
async def test_tool_mode_graph_memory_no_llm(storage, vector_store, mock_embed):
    gm = GraphMemory("u1", storage, vector_store, _raising_llm, mock_embed, tool_mode=True)
    await gm.memorize(
        "I work at Acme",
        triplets=[{"subject": "User", "predicate": "works_at", "object": "Acme"}],
    )
    results = await gm.retrieve("Where does the user work?", entities=["User"])
    assert any("User works_at Acme" in r.text for r in results)
