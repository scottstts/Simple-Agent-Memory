FILE_MEMORY_TOOL_INSTRUCTIONS = """File Memory

Purpose:
- Store and retrieve narrative facts: preferences, habits, schedules, goals, personal details.

Tools:
1) FileMemory.memorize (WRITE)
   When to use: new user facts, corrections, or updates that are narrative/atomic.
   Inputs:
   - text: original user utterance (string)
   - items: list of {content, category}
   Output:
   - list of stored MemoryItem objects (ids/metadata)

2) FileMemory.retrieve (READ)
   When to use: answer questions about preferences, schedules, personal info.
   Inputs:
   - query: user question
   - level: "summaries" | "items" | "resources" | "auto" | "vector_only" | "summaries_then_vector" | "items_then_vector" | "resources_then_vector"
   - categories: list of categories to load (optional)
   - search_query: keyword query (optional)
   Output:
   - formatted text block (summaries/items/resources + optional semantic items)

Guidance:
- Keep items atomic and explicit.
- Prefer these categories when they fit: work, preferences, personal, health, schedule, travel, tech, relationships.
- Summaries are produced by maintenance; retrieval returns both general and persistent summaries when available.
"""


GRAPH_MEMORY_TOOL_INSTRUCTIONS = """Graph Memory

Purpose:
- Store and retrieve relational facts as subject–predicate–object triplets.

Triplet schema:
{subject, predicate, object, status}
status = "current" | "past" | "uncertain"

Tools:
1) GraphMemory.memorize (WRITE)
   When to use: stable relations (works_at, manages, owns, located_in, etc.).
   Inputs:
   - text: original user utterance (string)
   - triplets: list of {subject, predicate, object, status}
   Output:
   - list of stored Triplet objects

2) GraphMemory.retrieve (READ)
   When to use: answer relationship/identity questions.
   Inputs:
   - query: user question
   - entities: list of entity names (required in tool_mode)
   - level: "graph_only" | "graph_then_vector" | "vector_only"
   Output:
   - list of RetrievalResult objects (text includes status label)

Guidance:
- Use clean entities for subject/object (names, orgs, roles).
- Avoid long narrative objects; split into multiple triplets.
- Use \"past\" for historical facts rather than deleting them.
- Graph retrieval annotates facts with status, e.g. \"(current)\" or \"(past)\".
- Current facts are prioritized over past facts.
"""
