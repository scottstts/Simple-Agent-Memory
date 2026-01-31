EVOLVE_SUMMARY = """You are a Memory Synchronization Specialist.

## Current Summary for "{category}"
{existing}

## New Memory Items to Integrate
{new_items}

## Task
1. If new items conflict with the current summary, overwrite the old facts with the new ones.
2. If items are new information, add them logically.
3. Remove redundant or outdated information.
4. Return ONLY the updated markdown summary. No preamble, no explanation."""

COMPRESS_MEMORIES = """Compress these memory items into a concise summary.
Preserve all important facts but eliminate redundancy.

Items:
{items}

Return a markdown summary that captures the essential information.
No preamble, no explanation — just the summary."""
