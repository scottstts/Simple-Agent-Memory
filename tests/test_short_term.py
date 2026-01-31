import pytest
from simple_agent_memory.short_term import ShortTermMemory


@pytest.mark.asyncio
async def test_save_and_load(storage):
    stm = ShortTermMemory("thread1", storage)
    await stm.save("step1", {"messages": ["hello"]})
    state = await stm.load_latest()
    assert state == {"messages": ["hello"]}


@pytest.mark.asyncio
async def test_rewind(storage):
    stm = ShortTermMemory("thread1", storage)
    await stm.save("s1", {"turn": 1})
    await stm.save("s2", {"turn": 2})
    await stm.save("s3", {"turn": 3})

    state = await stm.rewind("s1")
    assert state == {"turn": 1}


@pytest.mark.asyncio
async def test_list_steps(storage):
    stm = ShortTermMemory("thread1", storage)
    await stm.save("a", {"x": 1})
    await stm.save("b", {"x": 2})
    steps = await stm.list_steps()
    assert steps == ["a", "b"]


@pytest.mark.asyncio
async def test_empty_thread(storage):
    stm = ShortTermMemory("empty", storage)
    assert await stm.load_latest() is None
    assert await stm.rewind("nonexistent") is None
