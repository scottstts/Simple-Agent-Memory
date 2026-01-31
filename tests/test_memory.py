import pytest
from simple_agent_memory import Memory


@pytest.mark.asyncio
async def test_memory_file_mode(tmp_path, mock_llm):
    db = tmp_path / "mem.db"
    async with Memory("u1", mock_llm, db_path=db, mode="file") as mem:
        await mem.memorize("I prefer Python for scripting")
        result = await mem.retrieve("What language?")
        assert len(result) > 0


@pytest.mark.asyncio
async def test_memory_file_mode_with_embed_combines_semantic(tmp_path, mock_llm, mock_embed):
    db = tmp_path / "mem.db"
    async with Memory("u1", mock_llm, embed=mock_embed, db_path=db, mode="file") as mem:
        await mem.memorize("I prefer Python for scripting")
        result = await mem.retrieve("language preference")
        assert "RELEVANT MEMORIES" in result


@pytest.mark.asyncio
async def test_memory_graph_mode(tmp_path, mock_llm, mock_embed):
    db = tmp_path / "mem.db"
    async with Memory("u1", mock_llm, embed=mock_embed, db_path=db, mode="graph") as mem:
        await mem.memorize("I prefer Python and work at Acme")
        result = await mem.retrieve("What does the user prefer?")
        assert len(result) > 0


@pytest.mark.asyncio
async def test_memory_both_mode(tmp_path, mock_llm, mock_embed):
    db = tmp_path / "mem.db"
    async with Memory("u1", mock_llm, embed=mock_embed, db_path=db, mode="both") as mem:
        await mem.memorize("I prefer Python")
        result = await mem.retrieve("language preference")
        assert len(result) > 0


@pytest.mark.asyncio
async def test_checkpoint_via_memory(tmp_path, mock_llm):
    db = tmp_path / "mem.db"
    async with Memory("u1", mock_llm, db_path=db) as mem:
        cp = mem.checkpoint("thread1")
        await cp.save("s1", {"msg": "hello"})
        state = await cp.load_latest()
        assert state == {"msg": "hello"}


@pytest.mark.asyncio
async def test_maintain(tmp_path, mock_llm):
    db = tmp_path / "mem.db"
    async with Memory("u1", mock_llm, db_path=db) as mem:
        stats = await mem.maintain()
        assert "nightly" in stats


def test_graph_mode_requires_embed(tmp_path, mock_llm):
    with pytest.raises(ValueError, match="embed"):
        Memory("u1", mock_llm, db_path=tmp_path / "m.db", mode="graph")
