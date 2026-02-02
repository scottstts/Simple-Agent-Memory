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
   - level: "summaries" | "items" | "resources" | "auto" | "semantic" | "summaries_then_semantic" | "items_then_semantic" | "resources_then_semantic"
     (aliases: "vector_only" => "semantic", "*_then_vector" => "*_then_semantic")
   - categories: list of categories to load (optional)
   - search_query: keyword query (optional)
   Output:
   - formatted text block (summaries/items/resources + optional semantic items)

Guidance:
- Keep items atomic and explicit.
- Prefer these categories when they fit: work, preferences, personal, health, schedule, behavior, goals, travel, tech, relationships.
- For tool-mode retrieval, keep search_query short and keyword-like (2-8 terms). Avoid long sentences.
- If a keyword retrieval returns empty or misses details, retry with semantic retrieval (level "semantic" or "*_then_semantic").
- Summaries are produced by maintenance; retrieval returns both general and persistent summaries when available.
- Semantic retrieval returns a ranked memory block with timestamps and confidence.
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
   - expand: "none" | "low" | "medium" | "high" | "full" (controls connected-node expansion)
   Output:
   - list of RetrievalResult objects (text includes status label)

Guidance:
- Use clean entities for subject/object (names, orgs, roles).
- Avoid long narrative objects; split into multiple triplets.
- Use \"past\" for historical facts rather than deleting them.
- Graph retrieval annotates facts with status, e.g. \"(current)\" or \"(past)\".
- Current facts are prioritized over past facts.
- expand guidance: "none" = only direct subject triplets; "low" = include one hop with tight predicate filtering; "medium" = include one hop with expanded predicate filtering; "high" = one hop without predicate filtering; "full" = two hops without predicate filtering.
"""
