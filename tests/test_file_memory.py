import pytest
from simple_agent_memory.long_term.file_memory import FileMemory
from simple_agent_memory.maintenance import MaintenanceRunner


@pytest.mark.asyncio
async def test_memorize_creates_items_and_categories(storage, mock_llm):
    fm = FileMemory("u1", storage, mock_llm)
    items = await fm.memorize("I prefer Python for scripting and I work at Acme")

    assert len(items) == 2
    runner = MaintenanceRunner(storage, mock_llm)
    await runner.nightly("u1")
    categories = await storage.list_categories("u1")
    assert len(categories) >= 1


@pytest.mark.asyncio
async def test_retrieve_returns_content(storage, mock_llm):
    fm = FileMemory("u1", storage, mock_llm)
    await fm.memorize("I prefer Python for scripting")
    runner = MaintenanceRunner(storage, mock_llm)
    await runner.nightly("u1")

    result = await fm.retrieve("What language does the user prefer?")
    assert len(result) > 0


@pytest.mark.asyncio
async def test_retrieve_empty_returns_empty(storage, mock_llm):
    fm = FileMemory("u1", storage, mock_llm)
    result = await fm.retrieve("anything")
    assert result == ""
