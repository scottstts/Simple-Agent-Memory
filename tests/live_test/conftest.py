import os
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "live_test_mock_data"
LOG_DIR = BASE_DIR / "live_test_logs"


def _load_env() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _live_ready() -> bool:
    return os.getenv("RUN_LIVE_TESTS") == "1" and bool(os.getenv("OPENAI_API_KEY"))


def pytest_configure(config):
    config.addinivalue_line("markers", "live: live OpenAI integration tests")
    config.addinivalue_line("markers", "live_judge: live LLM judge evaluation")


_load_env()


def pytest_collection_modifyitems(session, config, items):
    judge_items = [i for i in items if i.get_closest_marker("live_judge")]
    other_items = [i for i in items if i not in judge_items]
    items[:] = other_items + judge_items


@pytest.fixture(autouse=True)
def _live_guard():
    if not _live_ready():
        pytest.skip("Live tests require RUN_LIVE_TESTS=1 and OPENAI_API_KEY")


@pytest.fixture
def live_log_dir(request):
    from uuid import uuid4

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    name = request.node.name.replace("/", "_")
    path = LOG_DIR / f"{name}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def openai_llm():
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - only for live envs
        pytest.skip(f"openai SDK not available: {exc}")

    client = OpenAI()

    def call(prompt: str) -> str:
        try:
            resp = client.responses.create(
                model="gpt-5.2",
                input=prompt,
                reasoning={"effort": "none"},
                temperature=1.0,
                max_output_tokens=5000,
            )
            return resp.output_text
        except Exception:
            resp = client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=5000,
            )
            return resp.choices[0].message.content or ""

    return call


@pytest.fixture
def openai_embedder():
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - only for live envs
        pytest.skip(f"openai SDK not available: {exc}")

    client = OpenAI()

    def embed(text: str) -> list[float]:
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding

    return embed


@pytest.fixture
def openai_judge():
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - only for live envs
        pytest.skip(f"openai SDK not available: {exc}")

    client = OpenAI()

    def call(system_prompt: str, user_prompt: str) -> str:
        try:
            resp = client.responses.create(
                model="gpt-5.2",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning={"effort": "medium"},
                temperature=1.0,
                max_output_tokens=1800,
            )
            return resp.output_text
        except Exception:
            resp = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                max_tokens=1800,
            )
            return resp.choices[0].message.content or ""

    return call
