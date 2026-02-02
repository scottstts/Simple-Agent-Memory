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

file_memory = FileMemory(
    user_id="user_123",
    storage=store,
    vector_store=vector,
    embed=embed,
    tool_mode=True,
)

graph_memory = GraphMemory(
    user_id="user_123",
    storage=store,
    vector_store=vector,
    embed=embed,
    tool_mode=True,
)
```

Your agent framework should expect agent tools to be **callable functions**, you can pass the bound methods:

```python
# register tool functions
write_file_memory_tool = file_memory.memorize
read_file_memory_tool = file_memory.retrieve
write_graph_memory_tool = graph_memory.memorize
read_graph_memory_tool = graph_memory.retrieve

# inject agent system prompt with memory tool-use instructions (out of the box)
# Or you can define tool-use instructions yourself if you want
instruction = (
    load_prompt("main_agent_system_prompt.md")
    .replace(
        "<memory_tool_instructions>",
        file_memory.tool_use_instruction + "\n\n" + graph_memory.tool_use_instruction,
    )
)

# define your agent and include the memory tools
agent = Agent(
    model="openai/gpt-5.2",
    tools=[
        write_file_memory_tool,
        read_file_memory_tool,
        write_graph_memory_tool,
        read_graph_memory_tool,
    ],
    instruction=instruction,
    # ...
)
```

### Tool-mode write/read (agent-controlled)

Below is an example **LLM tool call trace** (JSON-style) showing how an agent would call the tools and receive responses:

Note: file memory summaries are produced by maintenance (nightly general summary + weekly persistent summary). Tool calls should only pass atomic `items`.

```json
[
  {
    "role": "assistant",
    "tool_calls": [
      {
        "name": "FileMemory.memorize",
        "arguments": {
          "text": "I prefer Python for scripting and I work at Acme.",
          "items": [
            {"content": "User prefers Python for scripting", "category": "preferences"},
            {"content": "User works at Acme", "category": "work"}
          ]
        }
      }
    ]
  },
  {
    "role": "tool",
    "name": "FileMemory.memorize",
    "content": [{"id": "item_1"}, {"id": "item_2"}]
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "name": "FileMemory.retrieve",
        "arguments": {
          "query": "What language does the user prefer?",
          "level": "items",
          "search_query": "python"
        }
      }
    ]
  },
  {
    "role": "tool",
    "name": "FileMemory.retrieve",
    "content": "## Retrieved Items\n- User prefers Python for scripting"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "name": "GraphMemory.memorize",
        "arguments": {
          "text": "I work at Acme Corp.",
          "triplets": [
            {"subject": "User", "predicate": "works_at", "object": "Acme Corp", "status": "current"}
          ]
        }
      }
    ]
  },
  {
    "role": "tool",
    "name": "GraphMemory.memorize",
    "content": [{"subject": "User", "predicate": "works_at", "object": "Acme Corp", "status": "current"}]
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "name": "GraphMemory.retrieve",
        "arguments": {
          "query": "Where does the user work?",
          "entities": ["User"],
          "level": "graph_only"
        }
      }
    ]
  },
  {
    "role": "tool",
    "name": "GraphMemory.retrieve",
    "content": [
      {"text": "User works_at Acme Corp (current)", "score": 0.8}
    ]
  }
]
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
from simple_agent_memory import Maintenance
from simple_agent_memory.llm_clients import create_openai_client, create_openai_embedder

llm = create_openai_client("gpt-5.2")
embed = create_openai_embedder("text-embedding-3-small")

async with Maintenance("user_123", llm=llm, embed=embed, db_path="./agent_memory.db") as maint:
    await maint.run("nightly")   # dedup, merge, update summaries
    await maint.run("weekly")    # append persistent summaries, archive stale
    await maint.run("monthly")   # reindex embeddings
```

## Using File/Graph Memories Directly (Optional, Internal LLM Mode)

If you want the memory system itself to run extraction/classification (e.g., run `memorize()` manually for selective AI agent conversations, **not agent-driven**), use `FileMemory` and `GraphMemory` directly. This is optional and mainly for quick prototypes or non-tool workflows.

Note: this uses independent LLM chains running separate memory manipulations offline without the broader context your agent has, so expect some quality degradation compared to `tool_mode=True`.

```python
from simple_agent_memory import FileMemory, GraphMemory, Maintenance
from simple_agent_memory.storage import SQLiteStore
from simple_agent_memory.vector import NumpyVectorStore
from simple_agent_memory.llm_clients import create_openai_client, create_openai_embedder

llm = create_openai_client("gpt-5.2")
embed = create_openai_embedder("text-embedding-3-small")

store = SQLiteStore("./agent_memory.db")
vector = NumpyVectorStore("./agent_memory_vectors.db")

file_memory = FileMemory("user_123", storage=store, llm=llm, vector_store=vector, embed=embed)
graph_memory = GraphMemory("user_123", storage=store, vector_store=vector, llm=llm, embed=embed)

# Choose the appropriate memory per fact type.
await file_memory.memorize("I prefer Python for scripting.")
await graph_memory.memorize("I work at Acme Corp.")

async with Maintenance("user_123", llm=llm, db_path="./agent_memory.db", embed=embed) as maint:
    await maint.run("nightly")

context = await file_memory.retrieve("Write me a script", level="summaries")
print(context)
```

## Architecture

Simple Agent Memory exposes two long‑term memory systems (file memory and graph memory) plus a maintenance loop. The recommended and primary usage is **tool mode**: your agent calls the memory tools directly, and the memory system stores/retrieves exactly what the agent decides. The optional **internal LLM mode** runs separate LLM chains to extract/classify on your behalf. These modes are not used simultaneously.

### Tool Mode (recommended, agent‑driven)

**File memory (narrative facts)**
- **Inputs**: raw `text` plus agent‑provided atomic `items`.
- **Storage**:  
  - Every `text` is saved as a **resource** (immutable raw context).  
  - Each `item` is saved as a **memory item** (content + category + source link).  
  - If an embedder + vector store are configured, each item is embedded and indexed with metadata for semantic retrieval.
  - **Summaries are produced by maintenance**: a general per‑category summary (nightly) and a persistent per‑category summary (weekly).
- **Retrieval**: returns general + persistent summaries (clearly labeled), items, and/or raw resources based on the requested `level`. `search_query` can override the keyword used for item/resource lookups. Optional semantic context can be included via `semantic` or `*_then_semantic` (aliases: `vector_only`, `*_then_vector`). No internal LLM decisions are made.

**Graph memory (relational facts)**
- **Inputs**: raw `text` plus agent‑provided triplets `{subject, predicate, object, status}`.
- **Storage**:  
  - Every `text` is saved as a **resource**.  
  - Each triplet is saved to the **knowledge graph** with status (`current|past|uncertain`).  
  - The full `text` is embedded and indexed in the vector store (graph memory requires an embedder).
- **Retrieval**: graph lookup is driven by explicit `entities` (tool mode), with optional vector fallback/augmentation depending on `level` (`graph_only`, `graph_then_vector`, `vector_only`). You can also control connected-node expansion with `expand` (`none`, `low`, `medium`, `high`, `full`; default `medium`).

### Internal LLM Mode (optional, memory‑driven)

**File memory**
- **Inputs**: only raw `text`. The memory system uses its LLM to extract atomic facts and classify them into categories.
- **Retrieval**: the LLM selects relevant categories and decides whether summaries are sufficient; if not, it expands to items and resources. Summaries are still produced by maintenance (nightly/weekly).

**Graph memory**
- **Inputs**: only raw `text`. The memory system extracts triplets, detects conflicts for current facts, and deactivates prior conflicting predicates.
- **Retrieval**: the LLM extracts entities and may filter predicates; vector search can still be blended by `level`. Connected-node expansion is controlled by `expand` (`none`…`full`, default `medium`).

### Maintenance (system‑level, independent of mode)

Maintenance operates on **file memory items** to keep the store compact and high‑quality:
- **Nightly**: merges near‑duplicate items using embedding similarity, compresses them via LLM, updates embeddings/indexes, and removes merged items; also promotes frequently accessed items. It also rebuilds the **general** per‑category summaries from current items.
- **Weekly**: appends to the **persistent** per‑category summaries using older items and archives items not accessed recently (and removes them from the vector index).
- **Monthly**: re‑embeds all items, rebuilds the vector index, and archives very stale items.

### Both

If you want both memory types, call `FileMemory` and `GraphMemory` explicitly. This makes it clear which facts go into which memory and avoids accidental dual‑writes. Retrieval can still blend file summaries/items/resources with graph results by calling each retriever and composing the outputs at the agent layer.

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
