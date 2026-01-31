import pytest
from simple_agent_memory.retrieval import RetrievalPipeline
from simple_agent_memory.types import MemoryItem


@pytest.mark.asyncio
async def test_retrieve_with_items(storage, mock_llm):
    item = MemoryItem(user_id="u1", content="User prefers Python for scripting", category="preferences")
    await storage.save_item(item)

    pipeline = RetrievalPipeline(storage, None, mock_llm, None)
    result = await pipeline.retrieve("What scripting language?", "u1")

    assert "Python" in result
    assert "RELEVANT MEMORIES" in result


@pytest.mark.asyncio
async def test_retrieve_empty(storage, mock_llm):
    pipeline = RetrievalPipeline(storage, None, mock_llm, None)
    result = await pipeline.retrieve("anything", "u1")
    assert result == ""


@pytest.mark.asyncio
async def test_retrieve_respects_token_budget(storage, mock_llm):
    for i in range(20):
        item = MemoryItem(user_id="u1", content=f"Memory item number {i} " * 50, category="test")
        await storage.save_item(item)

    pipeline = RetrievalPipeline(storage, None, mock_llm, None, relevance_threshold=0.0)
    result = await pipeline.retrieve("memory", "u1", max_tokens=100)
    assert len(result) < 5000


@pytest.mark.asyncio
async def test_retrieve_updates_accessed_only_for_selected(storage):
    from datetime import datetime, timedelta, timezone

    def llm(_prompt: str) -> str:
        return "New"

    item_new = MemoryItem(user_id="u1", content="New " * 200, category="test")
    item_old = MemoryItem(user_id="u1", content="Old " * 200, category="test")
    now = datetime.now(timezone.utc)
    item_new.created_at = now
    item_new.accessed_at = now
    item_old.created_at = now - timedelta(days=10)
    item_old.accessed_at = now - timedelta(days=10)

    await storage.save_item(item_new)
    await storage.save_item(item_old)

    pipeline = RetrievalPipeline(storage, None, llm, None, relevance_threshold=0.0)
    await pipeline.retrieve("memory", "u1", max_tokens=300)

    updated_new = await storage.get_item_by_id(item_new.id)
    updated_old = await storage.get_item_by_id(item_old.id)

    assert updated_new is not None and updated_old is not None
    assert updated_new.access_count == 1
    assert updated_old.access_count == 0


@pytest.mark.asyncio
async def test_vector_retrieval_returns_text(storage, vector_store, mock_embed):
    from simple_agent_memory.retrieval import RetrievalPipeline

    def llm(_prompt: str) -> str:
        return "Vector text"

    embedding = mock_embed("Vector text")
    await vector_store.add(
        id="vec1",
        text="Vector text",
        embedding=embedding,
        metadata={"user_id": "u1"},
    )

    pipeline = RetrievalPipeline(storage, vector_store, llm, mock_embed, relevance_threshold=0.0)
    result = await pipeline.retrieve("irrelevant", "u1", max_tokens=200)
    assert "Vector text" in result
