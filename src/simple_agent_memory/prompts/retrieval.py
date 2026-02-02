GENERATE_QUERY = """Convert this user message into an optimized search query for retrieving relevant memories.
Focus on the key concepts, entities, and intent.
Include explicit times/dates/durations or conditional phrases if they appear in the message.

User message: {message}

Return ONLY the search query string, no quotes, no explanation."""

SUFFICIENCY_CHECK = """Given a user query and retrieved memory summaries, determine if the summaries contain enough information to fully answer the query.

Query: {query}

Retrieved summaries:
{summaries}

If the query asks for step-by-step routines, explicit times/dates, or fallback conditions and those details are missing, answer NO.
Answer with exactly YES or NO. Nothing else."""

GRAPH_PREDICATE_FILTER = """Given a user query and a list of predicates from a knowledge graph,
select which predicates are most relevant to answer the query.

Query: {query}
Available predicates:
{predicates}

Return a JSON array of predicate strings to include.
If unsure, include more rather than fewer.

Return ONLY valid JSON, no other text."""
