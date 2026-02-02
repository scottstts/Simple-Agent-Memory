CLASSIFY_ITEMS = """Classify each memory item into one of the existing categories, or suggest a new short category name.

Preferred categories (use these when they fit): work, preferences, personal, health, schedule, behavior, goals, travel, tech, relationships.

Existing categories:
{categories}

Items to classify:
{items}

Return a JSON array of objects with keys "content" and "category".
Use existing category names when appropriate. Only create new categories if none fit.
Keep category names lowercase, short, and descriptive (e.g., "work", "preferences", "health").

Rules of thumb:
- If an item mentions explicit times/dates or step-by-step routines, classify as "schedule".
- If an item describes a habit or repeated behavior, classify as "behavior".
- If an item expresses a desired change or plan, classify as "goals".
- If a contingency/fallback is tied to a routine, prefer "schedule" or "behavior".

Return ONLY valid JSON, no other text."""

SELECT_CATEGORIES = """Given a user query, select which memory categories are most likely to contain relevant information.

Query: {query}

Available categories:
{categories}

Return a JSON array of category name strings that are relevant to this query.
If unsure, include more rather than fewer.
For routine/schedule/habit/contingency questions, include "schedule", "behavior", and "goals" if available.

Return ONLY valid JSON, no other text."""
