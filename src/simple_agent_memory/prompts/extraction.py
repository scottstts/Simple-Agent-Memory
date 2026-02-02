EXTRACT_ITEMS = """Extract discrete, atomic facts explicitly stated in this conversation.
Focus on: preferences, behaviors/habits, personal details, opinions, goals, schedules/routines, and important statements.
Each fact should stand alone and be meaningful out of context.
Do NOT infer, assume, or reinterpret; only extract what is explicitly said.

Special handling for routines and schedules:
- Preserve exact times, dates, durations, and step order (e.g., "6:15 alarm", "6:30 run", "7:15 shower").
- If a routine has multiple steps, create a separate item for each step.
- If there is a conditional or fallback plan ("if it is pouring, do bodyweight workout"), extract it as its own item.
- Do not generalize away time-specific details.

Conversation:
{text}

Return a JSON array of objects with keys "content" and "category_hint".
category_hint should be a short label like "work", "preferences", "personal", "health", "goals", "schedule", "behavior", etc.
Use "schedule" for time-based routines or plans. Use "behavior" for habits. Use "goals" for desired changes/plans.

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
Use "past" for statements about previous roles/relationships (e.g., "I was at X") and for one-time events
(e.g., "hired", "founded", "met", "graduated") unless explicitly ongoing.
Use "current" for present-tense facts and ongoing relations (works_at, reports_to, lives_in, manages).
Use "uncertain" if unclear.

Normalization rules:
- Prefer canonical predicates (works_at, reports_to, manages, job_title, team, owns, located_in, spouse, parent_of).
- Avoid bespoke predicates derived from long clauses (e.g., "says_maybe_next_quarter_if").
- Do not emit multiple conflicting current triplets for the same subject+predicate.
- Only emit a "past" job_title if the text explicitly indicates it is former/previous; otherwise treat job_title as current.

Example output:
[
  {{"subject": "User", "predicate": "works_at", "object": "Acme Corp"}},
  {{"subject": "User", "predicate": "prefers", "object": "Python"}},
  {{"subject": "Acme Corp", "predicate": "industry", "object": "Technology"}}
]

Return ONLY valid JSON, no other text."""
