import pytest
from simple_agent_memory.maintenance import MaintenanceRunner
from simple_agent_memory.types import MemoryItem


@pytest.mark.asyncio
async def test_nightly_runs(storage, mock_llm):
    runner = MaintenanceRunner(storage, mock_llm)
    stats = await runner.nightly("u1")
    assert "merged" in stats
    assert "promoted" in stats


@pytest.mark.asyncio
async def test_weekly_archives_stale(storage, mock_llm):
    from datetime import datetime, timedelta, timezone
    old = MemoryItem(
        user_id="u1", content="Old fact", category="test",
    )
    old.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    old.accessed_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await storage.save_item(old)

    runner = MaintenanceRunner(storage, mock_llm)
    stats = await runner.weekly("u1")
    assert stats["archived"] >= 1


@pytest.mark.asyncio
async def test_run_all(storage, mock_llm):
    runner = MaintenanceRunner(storage, mock_llm)
    stats = await runner.run_all("u1")
    assert "nightly" in stats
    assert "weekly" in stats
    assert "monthly" in stats
