import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest


pytestmark = [pytest.mark.live, pytest.mark.live_judge]


LOG_ROOT = Path(__file__).parent / "live_test_logs"
DATA_ROOT = Path(__file__).parent / "live_test_mock_data"


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _db_snapshot(db_path: Path) -> dict:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    snapshot = {}

    try:
        cur.execute("SELECT category, summary FROM categories")
        snapshot["categories"] = [{"category": r[0], "summary": r[1]} for r in cur.fetchall()]
    except sqlite3.Error:
        snapshot["categories"] = []

    try:
        cur.execute(
            "SELECT content, category, created_at, accessed_at, archived FROM items ORDER BY created_at DESC LIMIT 100"
        )
        snapshot["items"] = [
            {
                "content": r[0],
                "category": r[1],
                "created_at": r[2],
                "accessed_at": r[3],
                "archived": bool(r[4]),
            }
            for r in cur.fetchall()
        ]
    except sqlite3.Error:
        snapshot["items"] = []

    try:
        cur.execute(
            "SELECT subject, predicate, object, timestamp, active, status FROM triplets ORDER BY timestamp DESC LIMIT 200"
        )
        snapshot["triplets"] = [
            {
                "subject": r[0],
                "predicate": r[1],
                "object": r[2],
                "timestamp": r[3],
                "active": bool(r[4]),
                "status": r[5],
            }
            for r in cur.fetchall()
        ]
    except sqlite3.Error:
        snapshot["triplets"] = []

    try:
        cur.execute("SELECT COUNT(*) FROM resources")
        snapshot["resource_count"] = cur.fetchone()[0]
    except sqlite3.Error:
        snapshot["resource_count"] = 0

    conn.close()
    return snapshot


def _collect_runs() -> list[dict]:
    if not LOG_ROOT.exists():
        return []

    runs = []
    for run_dir in sorted(LOG_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("test_live_judge_report"):
            continue
        run = {
            "name": run_dir.name,
            "path": str(run_dir),
            "retrieval_outputs": {},
            "maintenance": {},
            "db_snapshot": {},
            "meta": {},
        }
        for txt in run_dir.glob("*.txt"):
            run["retrieval_outputs"][txt.name] = _read_text(txt)
        for js in run_dir.glob("*.json"):
            if js.name == "maintenance_stats.json":
                try:
                    run["maintenance"] = json.loads(_read_text(js))
                except json.JSONDecodeError:
                    run["maintenance"] = {"raw": _read_text(js)}
            elif js.name == "run_meta.json":
                try:
                    run["meta"] = json.loads(_read_text(js))
                except json.JSONDecodeError:
                    run["meta"] = {"raw": _read_text(js)}

        db_path = run_dir / "memory.db"
        run["db_snapshot"] = _db_snapshot(db_path)
        runs.append(run)

    return runs


def _mock_context() -> dict:
    personal = json.loads(_read_text(DATA_ROOT / "personal_assistant_conversations.json"))
    graph = json.loads(_read_text(DATA_ROOT / "relationship_graph_conversations.json"))
    tool_mode = {
        "file_tool_mode": {
            "items": [
                {"content": "User prefers Python for scripting", "category": "preferences"},
                {"content": "User works at Acme Corp", "category": "work"},
            ],
        },
        "graph_tool_mode": {
            "triplets": [
                {"subject": "User", "predicate": "works_at", "object": "Acme Corp"}
            ]
        },
    }
    return {
        "personal_assistant": personal,
        "relationship_graph": graph,
        "tool_mode_inputs": tool_mode,
    }


@pytest.mark.asyncio
async def test_live_judge_report(openai_judge, live_log_dir):
    runs = _collect_runs()
    context = _mock_context()

    system_prompt = (
        "You are an expert evaluator for an AI memory system. "
        "You will be given: (1) mock conversation data (ground truth), "
        "(2) retrieval outputs, (3) database snapshots (categories, items, triplets), "
        "and (4) maintenance stats. "
        "Your job is to judge whether the memory system behaved reasonably and robustly.\n\n"
        "Important context about the system:\n"
        "- File-mode stores items; summaries are produced by maintenance and may include general + persistent views.\n"
        "- Vector retrieval may return raw conversation text (resources). This is expected and not hallucination.\n"
        "- Tool-mode uses pre-structured inputs; retrieval is query-driven and may NOT return all stored facts.\n"
        "- Graph-mode stores triplets and may also return vector hits; prioritize whether core relations are captured.\n\n"
        "Each run may include metadata (\"meta\") with the query used. Use that query intent when judging retrieval relevance.\n\n"
        "Scoring guidelines:\n"
        "- Focus on whether key facts from the mock data appear in stored summaries/items and retrieval outputs.\n"
        "- Penalize missing core facts, contradictions not resolved, or irrelevant retrievals.\n"
        "- For tool-mode, judge outputs against provided structured inputs and the query intent.\n"
        "- For graph-mode, check that major relations exist as triplets and that conflicts are handled sensibly.\n"
        "- Retrieval may be partial; do not penalize omission of unrelated facts.\n"
        "- Rate each run and an overall verdict.\n\n"
        "Return STRICT JSON with keys:\n"
        "{\n"
        "  \"overall_verdict\": \"pass|warn|fail\",\n"
        "  \"overall_score\": 0-100,\n"
        "  \"runs\": [\n"
        "    {\n"
        "      \"name\": \"...\",\n"
        "      \"score\": 0-100,\n"
        "      \"verdict\": \"pass|warn|fail\",\n"
        "      \"highlights\": [\"...\"],\n"
        "      \"issues\": [\"...\"]\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": \"...\"\n"
        "}"
    )

    payload = {
        "context": context,
        "runs": runs,
    }
    user_prompt = json.dumps(payload, indent=2)

    judge_output = openai_judge(system_prompt, user_prompt)
    out_dir = live_log_dir / f"judge_{uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge_input.json").write_text(user_prompt, encoding="utf-8")
    (out_dir / "judge_output.txt").write_text(judge_output, encoding="utf-8")

    try:
        report = json.loads(judge_output)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Judge output was not valid JSON: {exc}") from exc

    (out_dir / "judge_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    assert "overall_verdict" in report
