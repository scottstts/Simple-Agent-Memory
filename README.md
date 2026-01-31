# SAM (simple_agent_memory)

Simple, durable, plug and play memory system for agents.

- System level: short-term checkpoint based memory; periodic offline memory maintenance (consolidation, update, cleanup)

- Exposed to agent: long-term file memory and long-term graph memory

## Install

```bash
pip install git+ssh://git@github.com/scottstts/Simple-Agent-Memory.git
```

**With LLM provider support**
```bash
pip install "git+ssh://git@github.com/scottstts/Simple-Agent-Memory.git#egg=simple-agent-memory[openai]"
pip install "git+ssh://git@github.com/scottstts/Simple-Agent-Memory.git#egg=simple-agent-memory[anthropic]"
pip install "git+ssh://git@github.com/scottstts/Simple-Agent-Memory.git#egg=simple-agent-memory[gemini]"
pip install "git+ssh://git@github.com/scottstts/Simple-Agent-Memory.git#egg=simple-agent-memory[http]"
pip install "git+ssh://git@github.com/scottstts/Simple-Agent-Memory.git#egg=simple-agent-memory[all]"
```

## Recommended Usage: Tool Mode (Agent-Driven)

**Primary intended usage** is tool mode: your *agent* drives memory updates and retrievals. The memory system does not call its own LLM for extraction/classification — it simply stores and retrieves what the agent decides.

### Wire up tools

```python
from simple_agent_memory import FileMemory, GraphMemory
from simple_agent_memory.storage import SQLiteStore
from simple_agent_memory.vector import NumpyVectorStore
from simple_agent_memory.llm_clients import create_openai_embedder

embed = create_openai_embedder("text-embedding-3-small")

store = SQLiteStore("./agent_memory.db")
vector = NumpyVectorStore("./agent_memory_vectors.db")

file_memory_tool = FileMemory(
    user_id="user_123",
    storage=store,
    vector_store=vector,
    embed=embed,
    tool_mode=True,
)

graph_memory_tool = GraphMemory(
    user_id="user_123",
    storage=store,
    vector_store=vector,
    embed=embed,
    tool_mode=True,
)
```

### Inject tool instructions into your agent

```python
instruction = (
    load_prompt("main_agent_system_prompt.md")
    .replace(
        "<memory_tools>",
        file_memory_tool.tool_use_instruction + "\n\n" + graph_memory_tool.tool_use_instruction,
    )
)

agent = Agent(
    model="openai/gpt-5.2",
    tools=[file_memory_tool, graph_memory_tool],
    instruction=instruction,
    # ...
)
```

### Tool-mode write/read (agent-controlled)

```python
# File memory write: agent provides atomic items + summaries
await file_memory_tool.memorize(
    "I prefer Python for scripting and I work at Acme.",
    items=[
        {"content": "User prefers Python for scripting", "category": "preferences"},
        {"content": "User works at Acme", "category": "work"},
    ],
    summaries={
        "preferences": "- Prefers Python for scripting",
        "work": "- Works at Acme",
    },
)

# File memory read: agent chooses retrieval depth
context = await file_memory_tool.retrieve(
    "What language does the user prefer?",
    level="items",            # summaries | items | resources | auto
    search_query="python",    # optional keyword query
)

# Graph memory write: agent provides clean triplets
await graph_memory_tool.memorize(
    "I work at Acme Corp.",
    triplets=[{"subject": "User", "predicate": "works_at", "object": "Acme Corp", "status": "current"}],
)

# Graph memory read: agent chooses graph depth
results = await graph_memory_tool.retrieve(
    "Where does the user work?",
    entities=["User"],
    level="graph_only",        # graph_only | graph_then_vector | vector_only
)
```

## Tool Instructions (Out-of-the-box)

File and graph memory include **ready-to-use tool instructions** for your agent’s system prompt:

```python
from simple_agent_memory import FileMemory, GraphMemory

print(FileMemory.tool_use_instruction)
print(GraphMemory.tool_use_instruction)
```

These instructions explain when to use file vs graph memory, how to write facts, and how to use the retrieval hierarchy.

## System-Level (Only) Components

These are **system‑level tools** and should **not** be exposed to the agent directly:

- **Short‑term memory (checkpointing)**
- **Maintenance (dedup / summarize / archive / reindex)**

### Short-Term Memory (system-level)

```python
from simple_agent_memory import ShortTermMemory
from simple_agent_memory.storage import SQLiteStore

store = SQLiteStore("./agent_memory.db")
cp = ShortTermMemory("thread_abc", store)
await cp.save("step_1", {"messages": [{"role": "user", "content": "hi"}]})
state = await cp.load_latest()
```

### Maintenance (system-level)

```python
from simple_agent_memory.llm_clients import create_openai_client

llm = create_openai_client("gpt-5.2")

async with Memory("user_123", llm=llm) as mem:
    await mem.maintain("nightly")   # dedup, merge
    await mem.maintain("weekly")    # re-summarize, archive stale
    await mem.maintain("monthly")   # reindex embeddings
```

## Internal LLM Usage (Optional)

If you want the memory system itself to run extraction/classification/summarization (not agent-driven), you can use the `Memory` facade. This is optional and mainly for quick prototypes or non-tool workflows.

```python
from simple_agent_memory import Memory
from simple_agent_memory.llm_clients import create_openai_client, create_openai_embedder

llm = create_openai_client("gpt-5.2")
embed = create_openai_embedder("text-embedding-3-small")

async with Memory("user_123", llm=llm, embed=embed, mode="both") as mem:
    await mem.memorize("I prefer Python for scripting. I work at Acme Corp.")
    context = await mem.retrieve("Write me a script", file_level="summaries")
    print(context)
```

## Architecture

The library implements two long‑term memory architectures.

### File-Based Memory

A three‑layer hierarchy that mimics how humans categorize knowledge:

- **Resources** — Raw data. Immutable conversation logs.
- **Items** — Atomic facts extracted from resources.
- **Categories** — Evolving markdown summaries.

**Write path (internal LLM mode):** Ingest text → extract atomic facts → classify into categories → evolve summaries.

**Read path (hierarchical):** Summaries → items → raw resources.

### Graph-Based Memory

A hybrid structure combining vector similarity with knowledge graph precision:

- **Vector store** — Semantic discovery.
- **Knowledge graph** — Subject–predicate–object triplets with status (`current|past|uncertain`).

**Write path (internal LLM mode):** Extract triplets → detect replacements → mark old facts as `past_replaced` → store new triplets.

**Read path (hierarchical):** Graph‑only → graph+vector → vector‑only (agent‑controlled).

### Both

Runs file + graph in parallel. Retrieval can blend summaries + graph as you choose.

## LLM Clients

Every client provides sync and async variants. The library accepts any `(str) -> str` callable — these are convenience wrappers.

```python
from simple_agent_memory.llm_clients import (
    create_openai_client, create_openai_async_client,
    create_anthropic_client, create_anthropic_async_client,
    create_gemini_client, create_gemini_async_client,
    create_xai_client, create_xai_async_client,
    create_openrouter_client, create_openrouter_async_client,
    create_ollama_client, create_ollama_async_client,
    create_lmstudio_client, create_lmstudio_async_client,
)
```

## Embedding Clients

Graph memory and semantic retrieval require an embedder `(str) -> list[float]`.

```python
from simple_agent_memory.llm_clients import (
    create_openai_embedder, create_openai_async_embedder,
    create_gemini_embedder,
    create_ollama_embedder, create_ollama_async_embedder,
    create_lmstudio_embedder, create_lmstudio_async_embedder,
    create_openrouter_embedder, create_openrouter_async_embedder,
)
```

## Storage

This library stores memory in SQLite files (no separate markdown files).

Default paths:
- `~/.simple_agent_memory/memory.db`
- `~/.simple_agent_memory/vectors.db` (if embeddings used)

If you pass `db_path="./my_agent.db"`:
- `./my_agent.db`
- `./my_agent_vectors.db`

Both databases contain **file memory + graph memory** records:
- `memory.db`: resources, items, categories, triplets, checkpoints
- `*_vectors.db`: embeddings for file items and graph resources

## Using Components Directly

```python
from simple_agent_memory.storage import SQLiteStore
from simple_agent_memory.vector import NumpyVectorStore
from simple_agent_memory.long_term import FileMemory, GraphMemory
from simple_agent_memory.maintenance import MaintenanceRunner
from simple_agent_memory.short_term import ShortTermMemory
```

## Project Structure

```
simple_agent_memory/
├── __init__.py
├── memory.py
├── types.py
├── llm.py
├── short_term.py
├── retrieval.py
├── maintenance.py
├── llm_clients/
├── prompts/
├── long_term/
│   ├── file_memory.py
│   └── graph_memory.py
├── storage/
│   ├── base.py
│   └── sqlite_store.py
└── vector/
    ├── base.py
    └── numpy_store.py
```

## Tests

```bash
pip install "git+ssh://git@github.com/scottstts/Simple-Agent-Memory.git#egg=simple-agent-memory[dev]"
pytest tests/ -v

# Live tests (requires OPENAI_API_KEY and RUN_LIVE_TESTS=1)
RUN_LIVE_TESTS=1 pytest tests/live_test -v
```
