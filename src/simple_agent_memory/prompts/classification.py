CLASSIFY_ITEMS = """Classify each memory item into one of the existing categories, or suggest a new short category name.

Preferred categories (use these when they fit): work, preferences, personal, health, schedule, travel, tech, relationships.

Existing categories:
{categories}

Items to classify:
{items}

Return a JSON array of objects with keys "content" and "category".
Use existing category names when appropriate. Only create new categories if none fit.
Keep category names lowercase, short, and descriptive (e.g., "work", "preferences", "health").

Return ONLY valid JSON, no other text."""

SELECT_CATEGORIES = """Given a user query, select which memory categories are most likely to contain relevant information.

Query: {query}

Available categories:
{categories}

Return a JSON array of category name strings that are relevant to this query.
If unsure, include more rather than fewer.

Return ONLY valid JSON, no other text."""
