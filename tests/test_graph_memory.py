import pytest
from simple_agent_memory.long_term.graph_memory import GraphMemory


@pytest.mark.asyncio
async def test_memorize_creates_triplets(storage, vector_store, mock_llm, mock_embed):
    gm = GraphMemory("u1", storage, vector_store, mock_llm, mock_embed)
    triplets = await gm.memorize("I prefer Python and work at Acme")

    assert len(triplets) == 2
    stored = await storage.get_triplets("u1")
    assert len(stored) == 2


@pytest.mark.asyncio
async def test_retrieve_returns_results(storage, vector_store, mock_llm, mock_embed):
    gm = GraphMemory("u1", storage, vector_store, mock_llm, mock_embed)
    await gm.memorize("I prefer Python and work at Acme")

    results = await gm.retrieve("What does the user prefer?")
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_conflict_resolution(storage, vector_store, mock_llm, mock_embed):
    from tests.conftest import make_mock_llm

    conflict_llm = make_mock_llm({"replace": "YES"})
    gm = GraphMemory("u1", storage, vector_store, conflict_llm, mock_embed)

    await gm.memorize("I work at Google")
    await gm.memorize("I work at OpenAI")

    triplets = await storage.get_triplets("u1", subject="User")
    active_work = [t for t in triplets if t.predicate == "works_at"]
    assert len(active_work) >= 1
