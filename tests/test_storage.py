import pytest
from simple_agent_memory.types import Checkpoint, MemoryItem, Triplet


@pytest.mark.asyncio
async def test_resource_roundtrip(storage):
    rid = await storage.save_resource("user1", "Hello world")
    content = await storage.get_resource(rid)
    assert content == "Hello world"


@pytest.mark.asyncio
async def test_item_save_and_get(storage):
    item = MemoryItem(user_id="user1", content="Likes Python", category="preferences")
    await storage.save_item(item)
    items = await storage.get_items("user1", category="preferences")
    assert len(items) == 1
    assert items[0].content == "Likes Python"


@pytest.mark.asyncio
async def test_item_search(storage):
    await storage.save_item(MemoryItem(user_id="u1", content="Loves Python scripting", category="prefs"))
    await storage.save_item(MemoryItem(user_id="u1", content="Has a cat named Whiskers", category="personal"))
    results = await storage.search_items("u1", "Python")
    assert len(results) == 1
    assert "Python" in results[0].content


@pytest.mark.asyncio
async def test_item_search_multi_term(storage):
    await storage.save_item(MemoryItem(user_id="u2", content="Proposed morning plan: set a 6:15 alarm.", category="schedule"))
    await storage.save_item(MemoryItem(user_id="u2", content="Morgan currently wakes up around 7:30.", category="schedule"))
    results = await storage.search_items("u2", "morning routine 6:15 alarm run 7:30")
    assert len(results) >= 1
    contents = [r.content for r in results]
    assert any("6:15" in c for c in contents)


@pytest.mark.asyncio
async def test_category_roundtrip(storage):
    await storage.save_category("u1", "work", "Works at Acme Corp")
    summary = await storage.load_category("u1", "work")
    assert summary == "Works at Acme Corp"
    cats = await storage.list_categories("u1")
    assert "work" in cats


@pytest.mark.asyncio
async def test_checkpoint_roundtrip(storage):
    cp = Checkpoint(thread_id="t1", step_id="s1", state={"messages": ["hi"]})
    await storage.save_checkpoint(cp)
    latest = await storage.get_latest_checkpoint("t1")
    assert latest is not None
    assert latest.state == {"messages": ["hi"]}

    cp2 = Checkpoint(thread_id="t1", step_id="s2", state={"messages": ["hi", "hello"]})
    await storage.save_checkpoint(cp2)
    at_s1 = await storage.get_checkpoint_at_step("t1", "s1")
    assert at_s1.state == {"messages": ["hi"]}

    steps = await storage.list_checkpoint_steps("t1")
    assert steps == ["s1", "s2"]


@pytest.mark.asyncio
async def test_triplet_save_and_query(storage):
    t = Triplet(subject="User", predicate="works_at", object="Acme")
    await storage.save_triplet("u1", t)
    triplets = await storage.get_triplets("u1", subject="User")
    assert len(triplets) == 1
    assert triplets[0].object == "Acme"


@pytest.mark.asyncio
async def test_triplet_deactivate(storage):
    t = Triplet(subject="User", predicate="works_at", object="Google")
    await storage.save_triplet("u1", t)
    await storage.deactivate_triplet("u1", "User", "works_at")
    active = await storage.get_triplets("u1", subject="User")
    assert len(active) == 0


@pytest.mark.asyncio
async def test_delete_items(storage):
    item1 = MemoryItem(user_id="u1", content="Item 1", category="test")
    item2 = MemoryItem(user_id="u1", content="Item 2", category="test")
    await storage.save_item(item1)
    await storage.save_item(item2)
    await storage.delete_items([item1.id])
    remaining = await storage.get_items("u1")
    assert len(remaining) == 1
    assert remaining[0].id == item2.id
