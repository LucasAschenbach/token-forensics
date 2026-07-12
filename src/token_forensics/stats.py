from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .cli import classify_tool, human_bytes, human_int, iter_rollouts, read_events


def cumulative_statistics(home: Path) -> dict[str, Any]:
    models: dict[str, Counter[str]] = defaultdict(Counter)
    model_sessions: dict[str, set[str]] = defaultdict(set)
    tool_counts: Counter[str] = Counter()
    session_rows: list[dict[str, Any]] = []

    for path in iter_rollouts(home):
        model = "unknown"
        session_id = path.stem[-36:]
        cwd = ""
        session_usage: Counter[str] = Counter()
        session_models: Counter[str] = Counter()
        try:
            for event in read_events(path):
                payload = event.get("payload") or {}
                kind = payload.get("type")
                if event.get("type") == "session_meta":
                    session_id = str(payload.get("id") or payload.get("session_id") or session_id)
                    cwd = str(payload.get("cwd") or cwd)
                elif event.get("type") == "turn_context":
                    model = str(payload.get("model") or model)
                elif kind == "thread_settings_applied":
                    model = str((payload.get("thread_settings") or {}).get("model") or model)

                if kind in {"function_call", "custom_tool_call", "tool_search_call", "web_search_call", "image_generation_call"}:
                    fallback = str(payload.get("name") or kind.removesuffix("_call"))
                    name, _ = classify_tool(payload, fallback)
                    tool_counts[name] += 1

                usage = (payload.get("info") or {}).get("last_token_usage") if kind == "token_count" else None
                if usage:
                    current = models[model]
                    current["inferences"] += 1
                    current["input"] += int(usage.get("input_tokens") or 0)
                    current["cached"] += int(usage.get("cached_input_tokens") or 0)
                    current["output"] += int(usage.get("output_tokens") or 0)
                    current["reasoning"] += int(usage.get("reasoning_output_tokens") or 0)
                    model_sessions[model].add(session_id)
                    session_models[model] += 1
                    session_usage["inferences"] += 1
                    session_usage["input"] += int(usage.get("input_tokens") or 0)
                    session_usage["cached"] += int(usage.get("cached_input_tokens") or 0)
        except (OSError, ValueError):
            continue
        if session_usage:
            session_rows.append(
                {
                    "session_id": session_id,
                    "cwd": cwd,
                    "model": session_models.most_common(1)[0][0],
                    **session_usage,
                }
            )

    model_rows = []
    for model, values in models.items():
        model_rows.append(
            {
                "model": model,
                "sessions": len(model_sessions[model]),
                **values,
                "uncached": values["input"] - values["cached"],
            }
        )
    model_rows.sort(key=lambda row: row["input"], reverse=True)
    session_rows.sort(key=lambda row: row["input"], reverse=True)

    subagents = subagent_statistics(home)
    return {
        "models": model_rows,
        "sessions": session_rows,
        "tools": [{"name": name, "calls": calls} for name, calls in tool_counts.most_common()],
        "subagents": subagents,
    }


def subagent_statistics(home: Path) -> list[dict[str, Any]]:
    database = home / "state_5.sqlite"
    if not database.exists():
        return []
    query = """
        select e.parent_thread_id, e.child_thread_id,
               coalesce(p.model, 'unknown'), coalesce(c.model, 'unknown'),
               coalesce(c.tokens_used, 0), coalesce(c.cwd, '')
        from thread_spawn_edges e
        left join threads p on p.id = e.parent_thread_id
        left join threads c on c.id = e.child_thread_id
        order by e.parent_thread_id, e.child_thread_id
    """
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        rows = connection.execute(query).fetchall()
        connection.close()
    except sqlite3.Error:
        return []
    return [
        {
            "parent_id": row[0],
            "child_id": row[1],
            "parent_model": row[2],
            "child_model": row[3],
            "child_tokens": row[4],
            "cwd": row[5],
        }
        for row in rows
    ]


def statistics_lines(stats: dict[str, Any]) -> list[str]:
    models = stats["models"]
    total_input = sum(row["input"] for row in models)
    total_inferences = sum(row["inferences"] for row in models)
    total_sessions = len({row["session_id"] for row in stats["sessions"]})
    lines = [
        "CUMULATIVE STATISTICS",
        "",
        f"{total_sessions} sessions  {total_inferences:,} inferences  {total_input:,} submitted input tokens",
        "",
        "PER MODEL",
        "MODEL                 SESS   INFER      INPUT    AVG IN   CACHE  UNCACHED    OUTPUT    REASON",
    ]
    for row in models:
        average = row["input"] / max(1, row["inferences"])
        cache_rate = row["cached"] / max(1, row["input"]) * 100
        lines.append(
            f"{row['model'][:20]:20} {row['sessions']:5d} {row['inferences']:7d} {human_int(row['input']):>10} "
            f"{human_int(round(average)):>9} {cache_rate:6.1f}% {human_int(row['uncached']):>9} "
            f"{human_int(row['output']):>9} {human_int(row['reasoning']):>9}"
        )

    lines.extend(["", "TOP SESSIONS BY SUBMITTED INPUT", "SESSION        MODEL               INFER      INPUT   CACHE  WORKING DIRECTORY"])
    for row in stats["sessions"][:20]:
        cache_rate = row["cached"] / max(1, row["input"]) * 100
        lines.append(
            f"{row['session_id'][:13]:13} {row['model'][:19]:19} {row['inferences']:7d} "
            f"{human_int(row['input']):>10} {cache_rate:6.1f}%  {row['cwd']}"
        )

    lines.extend(["", "MOST REQUESTED TOOLS", "CALLS  TOOL"])
    for row in stats["tools"][:30]:
        lines.append(f"{row['calls']:5d}  {row['name']}")

    children = stats["subagents"]
    lines.extend(["", "SUBAGENTS", f"{len(children)} recorded child sessions"])
    if children:
        lines.append("PARENT         PARENT MODEL     CHILD          CHILD MODEL      CHILD TOKENS")
        for row in children:
            lines.append(
                f"{row['parent_id'][:13]:13} {row['parent_model'][:16]:16} {row['child_id'][:13]:13} "
                f"{row['child_model'][:16]:16} {human_int(row['child_tokens']):>12}"
            )
    else:
        lines.append("No parent-child thread edges found.")
    return lines
