FILE_MEMORY_TOOL_INSTRUCTIONS = """File Memory (tool_mode):

Use for preferences, habits, schedules, goals, personal facts, and narrative context.
Inputs:
- items: list of {content, category}
- summaries: dict {category: markdown summary}

Retrieval controls:
- level: "summaries" | "items" | "resources" | "auto"
- categories: list of category names to load
- search_query: keyword query for items/resources

Guidance:
- Keep items atomic and explicit.
- Summaries should be short, current, and conflict-resolved.
"""


GRAPH_MEMORY_TOOL_INSTRUCTIONS = """Graph Memory (tool_mode):

Use for relational facts: subject-predicate-object.
Triplet schema:
{subject, predicate, object, status}
status = "current" | "past" | "uncertain"

Guidance:
- Use clean entities for subject/object (names, orgs, roles).
- Avoid long narrative objects; split into multiple triplets.
- Use "past" for historical facts rather than deleting them.

Retrieval controls:
- level: "graph_only" | "graph_then_vector" | "vector_only"
- entities: list of entity names to retrieve from the graph

Notes:
- Graph retrieval annotates facts with status, e.g. "(current)" or "(past)".
- Current facts are prioritized over past facts.
"""
