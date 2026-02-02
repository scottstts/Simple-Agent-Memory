import pytest

from simple_agent_memory.long_term.file_memory import FileMemory
from simple_agent_memory.types import MemoryItem


@pytest.mark.asyncio
async def test_semantic_retrieve_with_items(storage, mock_llm):
    item = MemoryItem(user_id="u1", content="User prefers Python for scripting", category="preferences")
    await storage.save_item(item)

    fm = FileMemory("u1", storage, mock_llm)
    result = await fm.retrieve("What scripting language?", level="semantic")

    assert "Python" in result
    assert "RELEVANT MEMORIES" in result


@pytest.mark.asyncio
async def test_semantic_retrieve_empty(storage, mock_llm):
    fm = FileMemory("u1", storage, mock_llm)
    result = await fm.retrieve("anything", level="semantic")
    assert result == ""


@pytest.mark.asyncio
async def test_semantic_retrieve_includes_vector(storage, vector_store, mock_llm, mock_embed):
    embedding = mock_embed("Vector text")
    await vector_store.add(
        id="vec1",
        text="Vector text",
        embedding=embedding,
        metadata={"user_id": "u1", "type": "item", "category": "test"},
    )

    fm = FileMemory("u1", storage, mock_llm, vector_store=vector_store, embed=mock_embed)
    result = await fm.retrieve("irrelevant", level="semantic", search_query="Vector text")

    assert "Vector text" in result
