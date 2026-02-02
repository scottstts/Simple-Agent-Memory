import pytest

from simple_agent_memory.long_term.file_memory import FileMemory
from simple_agent_memory.long_term.graph_memory import GraphMemory


def _raising_llm(_prompt: str) -> str:
    raise AssertionError("LLM should not be called in tool_mode")


@pytest.mark.asyncio
async def test_tool_mode_file_memory_no_llm(storage):
    fm = FileMemory("u1", storage, llm=None, tool_mode=True)
    await fm.memorize(
        "I prefer Python for scripting and I work at Acme.",
        items=[
            {"content": "User prefers Python for scripting", "category": "preferences"},
            {"content": "User works at Acme", "category": "work"},
        ],
    )

    result = await fm.retrieve("irrelevant", search_query="Python", level="items")
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
