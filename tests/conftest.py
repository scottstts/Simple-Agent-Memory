import json
import pytest
from pathlib import Path

from simple_agent_memory.storage.sqlite_store import SQLiteStore
from simple_agent_memory.vector.numpy_store import NumpyVectorStore


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_memory.db"


@pytest.fixture
def tmp_vec_db(tmp_path):
    return tmp_path / "test_vectors.db"


@pytest.fixture
async def storage(tmp_db):
    store = SQLiteStore(tmp_db)
    yield store
    await store.close()


@pytest.fixture
async def vector_store(tmp_vec_db):
    store = NumpyVectorStore(str(tmp_vec_db))
    yield store
    await store.close()


def make_mock_llm(responses: dict[str, str] | None = None):
    """Creates a mock LLM that returns canned responses based on prompt keywords."""
    default_responses = {
        "Extract discrete": json.dumps([
            {"content": "User prefers Python", "category_hint": "preferences"},
            {"content": "User works at Acme", "category_hint": "work"},
        ]),
        "Classify each": json.dumps([
            {"content": "User prefers Python", "category": "preferences"},
            {"content": "User works at Acme", "category": "work"},
        ]),
        "Extract knowledge graph": json.dumps([
            {"subject": "User", "predicate": "prefers", "object": "Python"},
            {"subject": "User", "predicate": "works_at", "object": "Acme"},
        ]),
        "Extract entity": json.dumps(["User", "Python"]),
        "select which": json.dumps(["preferences"]),
        "enough information": "YES",
        "contradict": "NO",
        "replace": "NO",
        "Synchronization": "Updated profile with new information.",
        "Compress": "User prefers Python and works at Acme.",
        "Convert this user": "Python",
    }
    if responses:
        default_responses.update(responses)

    def mock_llm(prompt: str) -> str:
        for keyword, response in default_responses.items():
            if keyword.lower() in prompt.lower():
                return response
        return "OK"

    return mock_llm


def make_mock_embed(dim: int = 64):
    """Creates a deterministic mock embedding function."""
    import hashlib
    import numpy as np

    def mock_embed(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(float)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    return mock_embed


@pytest.fixture
def mock_llm():
    return make_mock_llm()


@pytest.fixture
def mock_embed():
    return make_mock_embed()
