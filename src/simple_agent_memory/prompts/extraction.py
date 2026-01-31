EXTRACT_ITEMS = """Extract discrete, atomic facts explicitly stated in this conversation.
Focus on: preferences, behaviors, personal details, opinions, goals, and important statements.
Each fact should stand alone and be meaningful out of context.
Do NOT infer, assume, or reinterpret; only extract what is explicitly said.

Conversation:
{text}

Return a JSON array of objects with keys "content" and "category_hint".
category_hint should be a short label like "work", "preferences", "personal", "health", "goals", etc.

Example output:
[
  {{"content": "User prefers Python for scripting", "category_hint": "preferences"}},
  {{"content": "User works at Acme Corp", "category_hint": "work"}}
]

Return ONLY valid JSON, no other text."""

EXTRACT_TRIPLETS = """Extract knowledge graph triplets (subject, predicate, object) explicitly stated in this text.
Focus on relationships, attributes, and factual connections.
Do NOT infer or assume facts beyond the text.
Keep subjects and objects short and entity-like (names, roles, organizations, places, concrete attributes).
Avoid long clause-like objects. If a statement contains multiple relations, split into multiple triplets.
Skip subjective, non-relational, or purely narrative statements (those belong in file memory, not graph memory).

Text:
{text}

Return a JSON array of objects with keys "subject", "predicate", "object", and "status".
status must be one of: "current", "past", or "uncertain".
Use "past" for statements about previous roles/relationships (e.g., "I was at X"),
and "current" for present facts. Use "uncertain" if unclear.

Example output:
[
  {{"subject": "User", "predicate": "works_at", "object": "Acme Corp"}},
  {{"subject": "User", "predicate": "prefers", "object": "Python"}},
  {{"subject": "Acme Corp", "predicate": "industry", "object": "Technology"}}
]

Return ONLY valid JSON, no other text."""
