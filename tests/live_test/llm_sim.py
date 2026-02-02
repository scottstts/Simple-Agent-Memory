import json
import re
from typing import Any

from simple_agent_memory.tool_instructions import (
    FILE_MEMORY_TOOL_INSTRUCTIONS,
    GRAPH_MEMORY_TOOL_INSTRUCTIONS,
)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = stripped.strip("`")
    if "\n" in stripped:
        stripped = stripped.split("\n", 1)[1]
    stripped = stripped.strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped


def _extract_json(text: str) -> Any:
    candidates = []
    cleaned = _strip_code_fence(text)
    candidates.append(cleaned)
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(cleaned[brace_start:brace_end + 1])
    bracket_start = cleaned.find("[")
    bracket_end = cleaned.rfind("]")
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        candidates.append(cleaned[bracket_start:bracket_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"LLM did not return valid JSON. Output was:\n{stripped_preview(cleaned)}")


def stripped_preview(text: str, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def llm_json(llm, prompt: str) -> Any:
    raw = llm(prompt)
    return _extract_json(raw)


def tool_mode_file_items(llm, conversation: str, user_name: str) -> list[dict]:
    prompt = (
        "You are simulating an agent using the FileMemory tool.\n\n"
        f"{FILE_MEMORY_TOOL_INSTRUCTIONS}\n\n"
        "Task: Extract atomic memory items suitable for FileMemory.memorize in tool mode.\n"
        "Return STRICT JSON only:\n"
        '{"items":[{"content":"...","category":"..."}]}\n\n'
        "Rules:\n"
        "- Use concise, atomic factual statements.\n"
        "- Preserve explicit times/dates/durations and step order for routines.\n"
        "- If a routine has multiple steps, create separate items for each step.\n"
        "- Extract conditional/fallback plans as their own items.\n"
        "- Use categories when appropriate (work, preferences, personal, health, schedule, travel, tech, relationships).\n"
        "- Use category 'schedule' for time-based routines; 'behavior' for habits; 'goals' for desired changes.\n"
        "- Encode corrections explicitly (e.g., \"User previously used X but now uses Y\").\n"
        "- Avoid duplicates.\n"
        "- At most 80 items.\n\n"
        f"User name: {user_name}\n"
        "Conversation transcript:\n"
        f"{conversation}\n"
    )
    data = llm_json(llm, prompt)
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("LLM did not return items list for tool-mode file memory.")
    return items


def tool_mode_graph_triplets(llm, conversation: str, user_name: str) -> list[dict]:
    prompt = (
        "You are simulating an agent using the GraphMemory tool.\n\n"
        f"{GRAPH_MEMORY_TOOL_INSTRUCTIONS}\n\n"
        "Task: Extract relational triplets for GraphMemory.memorize in tool mode.\n"
        "Return STRICT JSON only:\n"
        '{"triplets":[{"subject":"...","predicate":"...","object":"...","status":"current|past|uncertain"}]}\n\n'
        "Rules:\n"
        "- Use subject 'User' for the user.\n"
        "- Use clean entity names for subjects/objects (people, orgs, roles, places).\n"
        "- Use snake_case predicates (works_at, manages, lives_in, etc.).\n"
        "- If a relation changed, include old as status=\"past\" and new as status=\"current\".\n"
        "- Treat one-time events (hired, promoted, founded, met, graduated) as status=\"past\" unless explicitly ongoing.\n"
        "- Avoid multiple current triplets with the same subject+predicate.\n"
        "- At most 120 triplets.\n\n"
        f"User name: {user_name}\n"
        "Conversation transcript:\n"
        f"{conversation}\n"
    )
    data = llm_json(llm, prompt)
    triplets = data.get("triplets") if isinstance(data, dict) else data
    if not isinstance(triplets, list):
        raise ValueError("LLM did not return triplets list for tool-mode graph memory.")
    return triplets


def file_retrieval_plan(llm, conversation: str, tool_mode: bool) -> dict:
    mode_text = "tool_mode=True" if tool_mode else "tool_mode=False"
    prompt = (
        "You are preparing to call FileMemory.retrieve.\n\n"
        f"{FILE_MEMORY_TOOL_INSTRUCTIONS}\n\n"
        f"Return STRICT JSON only for {mode_text}:\n"
        '{"query":"...","level":"summaries|items|resources|semantic|summaries_then_semantic|items_then_semantic|resources_then_semantic","search_query":"...","categories":["..."]}\n\n'
        "Rules:\n"
        "- Create ONE realistic user question answerable from the transcript.\n"
        "- If tool_mode=True, include a helpful search_query (short keyword list, 2-8 terms). Categories optional.\n"
        "- If tool_mode=False, search_query and categories are optional.\n"
        "- Prefer level items_then_semantic for step-by-step routines or time-specific questions.\n"
        "- Omit categories unless you are very confident; if you include categories for routine/habit questions, include schedule + behavior + goals.\n"
        "- Avoid long search_query strings; if unsure, leave search_query empty and rely on semantic.\n"
        "- Keep query specific and useful.\n\n"
        "Conversation transcript:\n"
        f"{conversation}\n"
    )
    data = llm_json(llm, prompt)
    if not isinstance(data, dict) or "query" not in data:
        raise ValueError("LLM did not return a valid file retrieval plan.")
    return data


def graph_retrieval_plan(llm, conversation: str, tool_mode: bool) -> dict:
    mode_text = "tool_mode=True" if tool_mode else "tool_mode=False"
    prompt = (
        "You are preparing to call GraphMemory.retrieve.\n\n"
        f"{GRAPH_MEMORY_TOOL_INSTRUCTIONS}\n\n"
        f"Return STRICT JSON only for {mode_text}:\n"
        '{"query":"...","level":"graph_only|graph_then_vector|vector_only","entities":["..."],"expand":"none|low|medium|high|full"}\n\n'
        "Rules:\n"
        "- Create ONE realistic user question answerable from the transcript.\n"
        "- If tool_mode=True, include entities list required for graph retrieval.\n"
        "- For tool_mode=True, include 'User' plus any named entities needed to answer the query.\n"
        "- Use expand 'medium' for multi-hop reporting chain questions; use 'low' or 'none' for single-edge questions.\n"
        "- If tool_mode=False, entities can be omitted.\n"
        "- If unsure, set expand to 'medium'.\n\n"
        "Conversation transcript:\n"
        f"{conversation}\n"
    )
    data = llm_json(llm, prompt)
    if not isinstance(data, dict) or "query" not in data:
        raise ValueError("LLM did not return a valid graph retrieval plan.")
    return data


def both_mode_plan(llm, conversation: str) -> dict:
    prompt = (
        "You are preparing a combined retrieval plan for FileMemory and GraphMemory.\n\n"
        f"{FILE_MEMORY_TOOL_INSTRUCTIONS}\n\n"
        f"{GRAPH_MEMORY_TOOL_INSTRUCTIONS}\n\n"
        "Return STRICT JSON only:\n"
        '{"query":"...","file_level":"summaries|items|resources|semantic|summaries_then_semantic|items_then_semantic|resources_then_semantic","file_search_query":"...","graph_level":"graph_only|graph_then_vector|vector_only","graph_expand":"none|low|medium|high|full"}\n\n'
        "Rules:\n"
        "- Create ONE realistic user question answerable from the transcript.\n"
        "- file_search_query is optional; include if helpful.\n"
        "- If unsure, set graph_expand to 'medium'.\n\n"
        "Conversation transcript:\n"
        f"{conversation}\n"
    )
    data = llm_json(llm, prompt)
    if not isinstance(data, dict) or "query" not in data:
        raise ValueError("LLM did not return a valid combined retrieval plan.")
    return data
