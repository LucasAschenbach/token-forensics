from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from . import __version__


RESET = "\x1b[0m"
COLORS = {
    "dim": "\x1b[2m",
    "bold": "\x1b[1m",
    "cyan": "\x1b[36m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "red": "\x1b[31m",
    "magenta": "\x1b[35m",
}


@dataclass
class Usage:
    input: int = 0
    cached: int = 0
    output: int = 0
    reasoning: int = 0
    total: int = 0

    @property
    def uncached(self) -> int:
        return max(0, self.input - self.cached)

    def add(self, other: "Usage") -> None:
        self.input += other.input
        self.cached += other.cached
        self.output += other.output
        self.reasoning += other.reasoning
        self.total += other.total


@dataclass
class ToolCall:
    call_id: str
    name: str
    timestamp: str
    completed_at: str | None = None
    duration_ms: int | None = None
    success: bool | None = None
    result_bytes: int = 0
    title: str | None = None
    command: str | None = None
    workdir: str | None = None
    action: str | None = None
    arguments: dict[str, Any] | None = None


@dataclass
class Inference:
    index: int
    timestamp: str
    usage: Usage
    context_window: int | None
    tools: list[ToolCall] = field(default_factory=list)
    prior_tools: list[ToolCall] = field(default_factory=list)
    outputs: Counter[str] = field(default_factory=Counter)
    rate_primary: float | None = None
    rate_secondary: float | None = None
    primary_delta: float | None = None
    secondary_delta: float | None = None
    rate_gap_seconds: float | None = None
    rate_delta_attributable: bool = True
    rate_gap_start: str | None = None
    external_inferences: int = 0
    external_sessions: int = 0
    external_input_tokens: int = 0
    external_activity: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Session:
    path: str
    session_id: str = "unknown"
    cwd: str = "unknown"
    model: str = "unknown"
    effort: str = "unknown"
    started_at: str = ""
    inferences: list[Inference] = field(default_factory=list)
    notable_events: list[dict[str, str]] = field(default_factory=list)


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()


def iter_rollouts(home: Path) -> Iterable[Path]:
    for directory in (home / "sessions", home / "archived_sessions"):
        if directory.exists():
            yield from directory.rglob("rollout-*.jsonl")


def read_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                yield value


def event_kind(event: dict[str, Any]) -> str:
    payload = event.get("payload") or {}
    return str(payload.get("type") or event.get("type") or "unknown")


def json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
    except (TypeError, ValueError):
        return 0


def duration_ms(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return round(float(value) * 1000)
    if isinstance(value, dict):
        return round(value.get("secs", 0) * 1000 + value.get("nanos", 0) / 1_000_000)
    return None


def usage_from(payload: dict[str, Any]) -> Usage:
    raw = (payload.get("info") or {}).get("last_token_usage") or {}
    return Usage(
        input=int(raw.get("input_tokens") or 0),
        cached=int(raw.get("cached_input_tokens") or 0),
        output=int(raw.get("output_tokens") or 0),
        reasoning=int(raw.get("reasoning_output_tokens") or 0),
        total=int(raw.get("total_tokens") or 0),
    )


def tool_name(payload: dict[str, Any]) -> tuple[str, str | None]:
    invocation = payload.get("invocation") or {}
    arguments = invocation.get("arguments") or {}
    title = arguments.get("title") if isinstance(arguments, dict) else None
    server, tool = invocation.get("server"), invocation.get("tool")
    if server and tool:
        return f"{server}.{tool}", title
    return str(payload.get("name") or payload.get("tool_name") or "unknown_tool"), title


def parse_tool_input(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    raw: Any = payload.get("arguments")
    if raw is None:
        raw = payload.get("input")
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, dict):
            raw = decoded
    if isinstance(raw, dict):
        command = raw.get("cmd") or raw.get("command")
        workdir = raw.get("workdir") or raw.get("cwd")
        return (str(command) if command else None, str(workdir) if workdir else None)
    if not isinstance(raw, str):
        return None, None

    # Code-mode custom tools wrap exec_command in JavaScript. Decode its JSON string fields.
    def wrapped_field(name: str) -> str | None:
        match = re.search(rf"[\"']?\b{name}\b[\"']?\s*:\s*(\"(?:\\.|[^\"\\])*\")", raw)
        if not match:
            return None
        try:
            return str(json.loads(match.group(1)))
        except json.JSONDecodeError:
            return None

    return wrapped_field("cmd") or wrapped_field("command"), wrapped_field("workdir") or wrapped_field("cwd")


def summarize_command(command: str | None) -> str | None:
    if not command:
        return None
    paths = re.findall(r"['\"](/[^'\"]+)['\"]", command)
    if paths and ("sed -n" in command or re.search(r"(?:^|[;&|]\s*)cat\s", command)):
        labels = []
        for value in paths:
            path = Path(value)
            labels.append(f"{path.parent.name}/{path.name}")
        unique = list(dict.fromkeys(labels))
        return f"read {len(unique)} file{'s' if len(unique) != 1 else ''}: " + ", ".join(unique)
    return None


def classify_tool(payload: dict[str, Any], fallback: str) -> tuple[str, str | None]:
    if fallback == "spawn_agent":
        arguments = normalized_arguments(payload.get("arguments")) or {}
        target_model = arguments.get("model") or "default model"
        role = arguments.get("agent_type") or arguments.get("role") or "subagent"
        return "agent.spawn", f"spawn {role} using {target_model}"
    raw = payload.get("input")
    if not isinstance(raw, str):
        return fallback, None

    schema = re.search(r"ALL_TOOLS\.find\([^\n]*?name\s*===?\s*[\"']([^\"']+)", raw)
    if schema:
        target = schema.group(1).removeprefix("mcp__").replace("__", ".")
        return "tool_schema.inspect", f"load tool definition: {target}"
    if "ALL_TOOLS.filter" in raw:
        family = "XcodeBuildMCP" if re.search(r"xcode|simulator|build_run|list_sims", raw, re.I) else "available tools"
        return "tool_catalog.search", f"search tool catalog: {family}"

    match = re.search(r"tools\.([A-Za-z0-9_]+)\s*\(", raw)
    if not match:
        return fallback, None
    wrapped = match.group(1)
    if wrapped.startswith("mcp__"):
        label = wrapped.removeprefix("mcp__").replace("__", ".")
        return label, f"invoke MCP tool: {label}"
    mappings = {
        "exec_command": ("shell.exec", None),
        "apply_patch": ("apply_patch", "edit files with patch"),
        "view_image": ("image.view", "inspect local image"),
        "web__run": ("web.run", "search or inspect web sources"),
    }
    return mappings.get(wrapped, (wrapped.replace("__", "."), f"invoke tool: {wrapped.replace('__', '.')}"))


def normalized_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def seconds_between(previous: str | None, current: str) -> float | None:
    if not previous or not current:
        return None
    try:
        before = datetime.fromisoformat(previous.replace("Z", "+00:00"))
        after = datetime.fromisoformat(current.replace("Z", "+00:00"))
        return max(0.0, (after - before).total_seconds())
    except ValueError:
        return None


def parse_session(path: Path) -> Session:
    session = Session(path=str(path))
    buffer: list[dict[str, Any]] = []
    pending_prior_tools: list[ToolCall] = []
    previous_primary: float | None = None
    previous_secondary: float | None = None
    previous_rate_timestamp: str | None = None

    for event in read_events(path):
        payload = event.get("payload") or {}
        kind = event_kind(event)

        if event.get("type") == "session_meta":
            session.session_id = str(payload.get("id") or session.session_id)
            session.cwd = str(payload.get("cwd") or session.cwd)
            session.started_at = str(payload.get("timestamp") or event.get("timestamp") or "")
        elif event.get("type") == "turn_context":
            session.model = str(payload.get("model") or session.model)
            session.effort = str(payload.get("effort") or session.effort)

        if kind in {"context_compacted", "turn_aborted", "thread_rolled_back", "task_started", "task_complete", "user_message"}:
            session.notable_events.append({"timestamp": str(event.get("timestamp") or ""), "kind": kind})

        if kind != "token_count" or not (payload.get("info") or {}).get("last_token_usage"):
            buffer.append(event)
            continue

        calls: dict[str, ToolCall] = {}
        completions: dict[str, dict[str, Any]] = {}
        inner_mcp_completions: list[dict[str, Any]] = []
        outputs: Counter[str] = Counter()
        for buffered in buffer:
            item = buffered.get("payload") or {}
            item_kind = event_kind(buffered)
            if item_kind in {"reasoning", "message", "agent_message", "agent_reasoning"}:
                outputs[item_kind] += 1
            if item_kind in {"function_call", "custom_tool_call", "tool_search_call", "web_search_call", "image_generation_call"}:
                call_id = str(item.get("call_id") or item.get("id") or f"call-{len(calls)}")
                command, workdir = parse_tool_input(item)
                fallback_name = str(item.get("name") or item_kind.removesuffix("_call"))
                classified_name, classified_action = classify_tool(item, fallback_name)
                calls[call_id] = ToolCall(
                    call_id=call_id,
                    name=classified_name,
                    timestamp=str(buffered.get("timestamp") or ""),
                    command=command,
                    workdir=workdir,
                    action=classified_action or summarize_command(command),
                    arguments=normalized_arguments(item.get("arguments")),
                )
            if item_kind == "mcp_tool_call_end":
                inner_mcp_completions.append(buffered)
            if item_kind.endswith("_end") or item_kind in {"function_call_output", "custom_tool_call_output", "tool_search_output"}:
                call_id = str(item.get("call_id") or "")
                existing_kind = event_kind(completions[call_id]) if call_id in completions else ""
                if not existing_kind.endswith("_end") or item_kind.endswith("_end"):
                    completions[call_id] = buffered

        # Code-mode uses an outer custom-tool call ID and a different inner MCP ID.
        # Match the inner completion by its normalized server/tool identity.
        for completed in inner_mcp_completions:
            end = completed.get("payload") or {}
            invocation = end.get("invocation") or {}
            identity = f"{invocation.get('server')}.{invocation.get('tool')}"
            match = next((call for call in reversed(list(calls.values())) if call.name == identity), None)
            if match:
                completions[match.call_id] = completed

        completed_tools: list[ToolCall] = []
        for call_id, call in calls.items():
            completed = completions.get(call_id)
            if completed:
                end = completed.get("payload") or {}
                invocation = end.get("invocation") or {}
                resolved_name, title = tool_name(end)
                if resolved_name != "unknown_tool":
                    call.name = resolved_name
                call.title = title
                if isinstance(invocation.get("arguments"), dict):
                    call.arguments = invocation["arguments"]
                call.completed_at = str(completed.get("timestamp") or "")
                call.duration_ms = duration_ms(end.get("duration"))
                call.success = end.get("success")
                if call.success is None and end.get("status") is not None:
                    call.success = str(end.get("status")).lower() in {"ok", "success", "completed"}
                result = end.get("result", end.get("output", end.get("stdout", "")))
                call.result_bytes = json_size(result)
            completed_tools.append(call)

        rates = payload.get("rate_limits") or {}
        primary = (rates.get("primary") or {}).get("used_percent")
        secondary = (rates.get("secondary") or {}).get("used_percent")
        info = payload.get("info") or {}
        event_timestamp = str(event.get("timestamp") or "")
        rate_gap = seconds_between(previous_rate_timestamp, event_timestamp)
        rate_delta_attributable = rate_gap is None or rate_gap <= 300
        inference = Inference(
            index=len(session.inferences) + 1,
            timestamp=event_timestamp,
            usage=usage_from(payload),
            context_window=info.get("model_context_window"),
            tools=completed_tools,
            prior_tools=pending_prior_tools,
            outputs=outputs,
            rate_primary=primary,
            rate_secondary=secondary,
            primary_delta=(primary - previous_primary) if primary is not None and previous_primary is not None and primary >= previous_primary else None,
            secondary_delta=(secondary - previous_secondary) if secondary is not None and previous_secondary is not None and secondary >= previous_secondary else None,
            rate_gap_seconds=rate_gap,
            rate_delta_attributable=rate_delta_attributable,
            rate_gap_start=previous_rate_timestamp,
        )
        score_inference(inference)
        session.inferences.append(inference)
        pending_prior_tools = completed_tools
        if primary is not None:
            previous_primary = primary
        if secondary is not None:
            previous_secondary = secondary
        if primary is not None or secondary is not None:
            previous_rate_timestamp = event_timestamp
        buffer = []

    return session


def correlate_external_activity(session: Session, home: Path) -> None:
    candidates = [
        step for step in session.inferences
        if not step.rate_delta_attributable and step.rate_gap_start and ((step.primary_delta or 0) > 0 or (step.secondary_delta or 0) > 0)
    ]
    if not candidates:
        return
    source = Path(session.path).resolve()
    activity: dict[int, dict[str, dict[str, Any]]] = {step.index: {} for step in candidates}
    for path in iter_rollouts(home):
        if path.resolve() == source:
            continue
        path_key = str(path)
        metadata = {
            "session_id": path.stem[-36:],
            "cwd": "",
            "path": path_key,
            "inferences": 0,
            "input_tokens": 0,
        }
        try:
            for event in read_events(path):
                payload = event.get("payload") or {}
                timestamp = str(event.get("timestamp") or "")
                if event.get("type") == "session_meta":
                    metadata["session_id"] = str(payload.get("id") or payload.get("session_id") or metadata["session_id"])
                    metadata["cwd"] = str(payload.get("cwd") or "")
                if payload.get("type") != "token_count" or not (payload.get("info") or {}).get("last_token_usage"):
                    continue
                for step in candidates:
                    if step.rate_gap_start < timestamp < step.timestamp:
                        entry = activity[step.index].setdefault(path_key, dict(metadata))
                        step.external_inferences += 1
                        usage = usage_from(payload)
                        step.external_input_tokens += usage.input
                        entry["inferences"] += 1
                        entry["input_tokens"] += usage.input
        except (OSError, ValueError):
            continue
    for step in candidates:
        step.external_activity = sorted(activity[step.index].values(), key=lambda item: item["input_tokens"], reverse=True)
        step.external_sessions = len(step.external_activity)
        if step.external_inferences:
            step.warnings = [warning for warning in step.warnings if warning != "rate refresh after gap"]
            step.warnings.append("rate change includes other sessions")


def score_inference(step: Inference) -> None:
    usage = step.usage
    if usage.uncached >= 16_000:
        step.warnings.append("high uncached input")
    elif usage.uncached >= 4_000:
        step.warnings.append("uncached input")
    if usage.reasoning >= 4_000:
        step.warnings.append("heavy reasoning")
    elif usage.reasoning >= 1_500:
        step.warnings.append("reasoning")
    if any(tool.result_bytes >= 100_000 for tool in step.tools):
        step.warnings.append("huge tool result")
    elif any(tool.result_bytes >= 25_000 for tool in step.tools):
        step.warnings.append("large tool result")
    if any(tool.success is False for tool in step.tools):
        step.warnings.append("tool failure")
    if (step.primary_delta or 0) >= 2 or (step.secondary_delta or 0) >= 2:
        step.warnings.append("rate jump observed" if step.rate_delta_attributable else "rate refresh after gap")
    if step.rate_gap_seconds and step.rate_gap_seconds > 300 and usage.input and usage.cached / usage.input < 0.5:
        step.warnings.append("cold cache after idle")


def resolve_session(value: str, home: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate
    matches = [path for path in iter_rollouts(home) if value.lower() in path.name.lower()]
    if not matches:
        raise FileNotFoundError(f"no rollout matching {value!r} under {home}")
    if len(matches) > 1:
        choices = "\n".join(f"  {path}" for path in sorted(matches)[-10:])
        raise ValueError(f"session fragment is ambiguous ({len(matches)} matches):\n{choices}")
    return matches[0]


def compact_timestamp(value: str) -> str:
    if not value:
        return "--:--:--"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%H:%M:%S.%f")[:-3]
    except ValueError:
        return value[:12]


def human_int(value: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}m"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def human_bytes(value: int) -> str:
    if value >= 1_048_576:
        return f"{value / 1_048_576:.1f}MiB"
    if value >= 1024:
        return f"{value / 1024:.1f}KiB"
    return f"{value}B"


class Paint:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, color: str) -> str:
        return f"{COLORS[color]}{text}{RESET}" if self.enabled else text


def severity(step: Inference) -> str:
    critical = {"high uncached input", "huge tool result", "tool failure", "rate jump observed"}
    return "red" if critical.intersection(step.warnings) else "yellow" if step.warnings else "green"


def display_name(tool: ToolCall) -> str:
    return f"{tool.name} ({tool.title})" if tool.title else tool.name


def render_session(session: Session, color: bool, show_all_events: bool = True) -> None:
    p = Paint(color)
    print(p(f"SESSION {session.session_id}", "bold"))
    print(f"{p('model', 'dim')} {session.model}/{session.effort}  {p('cwd', 'dim')} {session.cwd}")
    print(f"{p('rollout', 'dim')} {session.path}")

    render_summary(session, p)
    print("\n" + p("EVENT TIMELINE", "bold"))
    print(p(timeline_header(), "dim"))

    notable = iter(sorted(session.notable_events, key=lambda item: item["timestamp"]))
    next_notable = next(notable, None)
    for step in session.inferences:
        while show_all_events and next_notable and next_notable["timestamp"] <= step.timestamp:
            marker_color = "magenta" if next_notable["kind"] in {"context_compacted", "turn_aborted", "thread_rolled_back"} else "cyan"
            print(f"{compact_timestamp(next_notable['timestamp'])}    {p('·', marker_color)}  {p(next_notable['kind'], marker_color)}")
            next_notable = next(notable, None)
        context_pct = (step.usage.input / step.context_window * 100) if step.context_window else 0
        rate_bits = []
        if step.rate_primary is not None:
            marker = "?" if step.primary_delta and not step.rate_delta_attributable else ""
            rate_bits.append(f"P{step.rate_primary:g}%" + (f"+{step.primary_delta:g}{marker}" if step.primary_delta else ""))
        if step.rate_secondary is not None:
            marker = "?" if step.secondary_delta and not step.rate_delta_attributable else ""
            rate_bits.append(f"S{step.rate_secondary:g}%" + (f"+{step.secondary_delta:g}{marker}" if step.secondary_delta else ""))
        warning = ", ".join(step.warnings) or "inference"
        row = (
            f"{compact_timestamp(step.timestamp)} {step.index:4d} "
            f"{human_int(step.usage.input):>7} {human_int(step.usage.cached):>7} "
            f"{human_int(step.usage.uncached):>7} {human_int(step.usage.output):>7} "
            f"{human_int(step.usage.reasoning):>7} {context_pct:5.1f}% "
            f"{'/'.join(rate_bits):<9} {warning}"
        )
        print(p(row, severity(step)))
        for tool in step.tools:
            detail = [display_name(tool)]
            if tool.duration_ms is not None:
                detail.append(f"{tool.duration_ms}ms")
            if tool.result_bytes:
                detail.append(f"result {human_bytes(tool.result_bytes)}")
            if tool.success is False:
                detail.append("FAILED")
            print(f"{'':17}{p('└─ next tool →', 'dim')} {p('  '.join(detail), 'red' if tool.success is False or tool.result_bytes >= 100_000 else 'cyan')}")
        if step.prior_tools:
            prior = []
            for tool in step.prior_tools:
                result = f" result {human_bytes(tool.result_bytes)}" if tool.result_bytes else ""
                prior.append(f"{display_name(tool)}{result}")
            print(f"{'':17}{p('↳ input from ←', 'dim')} {', '.join(prior)}")



def timeline_header() -> str:
    return (
        f"{'TIME':12} {'#':>4} {'INPUT':>7} {'CACHE':>7} {'UNCACH':>7} "
        f"{'OUTPUT':>7} {'REASON':>7} {'CTX':>6} {'RATE':<9} EVENT / SIGNAL"
    )


def render_summary(session: Session, p: Paint) -> None:
    total = Usage()
    for step in session.inferences:
        total.add(step.usage)
    print("\n" + p("TOTALS", "bold"))
    print(
        f"{len(session.inferences)} inferences  input {human_int(total.input)}  cached {human_int(total.cached)}  "
        f"uncached {human_int(total.uncached)}  output {human_int(total.output)}  reasoning {human_int(total.reasoning)}"
    )

    aggregates: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "usage": Usage(), "bytes": 0, "failures": 0})
    for step in session.inferences:
        for name in set(tool.name for tool in step.tools):
            aggregate = aggregates[name]
            aggregate["count"] += sum(tool.name == name for tool in step.tools)
            aggregate["usage"].add(step.usage)
            aggregate["bytes"] += sum(tool.result_bytes for tool in step.tools if tool.name == name)
            aggregate["failures"] += sum(tool.success is False for tool in step.tools if tool.name == name)
    if aggregates:
        print("\n" + p("BY REQUESTED TOOL", "bold") + p("  (inference usage; categories can overlap)", "dim"))
        print(p("CALLS  INPUT   CACHE  UNCACH  OUTPUT  REASON  RESULTS  TOOL", "dim"))
        for name, aggregate in sorted(aggregates.items(), key=lambda item: item[1]["usage"].input, reverse=True):
            usage = aggregate["usage"]
            line = (
                f"{aggregate['count']:5d} {human_int(usage.input):>7} {human_int(usage.cached):>7} "
                f"{human_int(usage.uncached):>7} {human_int(usage.output):>7} {human_int(usage.reasoning):>7} "
                f"{human_bytes(aggregate['bytes']):>8}  {name}"
            )
            print(p(line, "red" if aggregate["failures"] else "yellow" if usage.uncached >= 16_000 else "cyan"))

    suspects = sorted(session.inferences, key=lambda step: (len(step.warnings), step.usage.input, step.usage.uncached), reverse=True)[:5]
    suspects = [step for step in suspects if step.warnings]
    if suspects:
        print("\n" + p("TOP SUSPECTS", "bold"))
        for step in suspects:
            tools = ", ".join(tool.name for tool in step.tools) or "no tool"
            print(p(f"#{step.index} {compact_timestamp(step.timestamp)}  {', '.join(step.warnings)}  [{tools}]", severity(step)))


def session_to_dict(session: Session) -> dict[str, Any]:
    result = asdict(session)
    for inference in result["inferences"]:
        # dataclasses.asdict reconstructs Counter with tuple keys; JSON needs a plain map.
        inference["outputs"] = dict(session.inferences[inference["index"] - 1].outputs)
    return result


def session_preview(path: Path) -> dict[str, Any]:
    preview = {"path": str(path), "id": path.stem.rsplit("-", 5)[-5:]}
    preview["id"] = "-".join(preview["id"])
    preview["mtime"] = path.stat().st_mtime
    try:
        for event in read_events(path):
            if event.get("type") == "session_meta":
                payload = event.get("payload") or {}
                preview.update(id=payload.get("id", preview["id"]), cwd=payload.get("cwd", ""), started=payload.get("timestamp", ""))
                break
    except (OSError, ValueError):
        preview["cwd"] = "<unreadable>"
    return preview


def command_list(args: argparse.Namespace) -> int:
    paths = sorted(iter_rollouts(args.codex_home), key=lambda path: path.stat().st_mtime, reverse=True)
    previews = [session_preview(path) for path in paths]
    if args.query:
        query = args.query.lower()
        previews = [item for item in previews if query in json.dumps(item).lower()]
    if args.json:
        json.dump(previews[: args.limit], sys.stdout, indent=2)
        print()
        return 0
    p = Paint(args.color)
    print(p("STARTED                   SESSION                               CWD", "dim"))
    for item in previews[: args.limit]:
        print(f"{str(item.get('started', ''))[:25]:25} {str(item['id']):36} {item.get('cwd', '')}")
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    path = resolve_session(args.session, args.codex_home)
    session = parse_session(path)
    correlate_external_activity(session, args.codex_home)
    if args.json:
        json.dump(session_to_dict(session), sys.stdout, indent=2)
        print()
    else:
        render_session(session, args.color, not args.inferences_only)
    return 0


def command_tui(args: argparse.Namespace) -> int:
    from .tui import run_tui

    run_tui(args.codex_home)
    return 0


def command_stats(args: argparse.Namespace) -> int:
    from .stats import cumulative_statistics, statistics_lines

    statistics = cumulative_statistics(args.codex_home)
    if args.json:
        json.dump(statistics, sys.stdout, indent=2)
        print()
    else:
        print("\n".join(statistics_lines(statistics)))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="token-forensics", description="Analyze Codex token usage by inference and tool loop.")
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    root.add_argument("--codex-home", type=Path, default=codex_home(), help="Codex state directory (default: %(default)s)")
    root.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="terminal color policy")
    subparsers = root.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="list locally available sessions")
    listing.add_argument("query", nargs="?", help="filter by ID, path, or working directory")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(handler=command_list)

    analyze = subparsers.add_parser("analyze", help="analyze one rollout path or unique session-ID fragment")
    analyze.add_argument("session")
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--inferences-only", action="store_true", help="hide user/task/compaction markers")
    analyze.set_defaults(handler=command_analyze)

    tui = subparsers.add_parser("tui", help="open the interactive session browser")
    tui.set_defaults(handler=command_tui)

    stats = subparsers.add_parser("stats", help="show cumulative model, session, tool, and subagent statistics")
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(handler=command_stats)
    return root


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if not effective_argv:
        effective_argv = ["tui" if sys.stdin.isatty() and sys.stdout.isatty() else "list"]
    args = parser().parse_args(effective_argv)
    args.color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty() and not getattr(args, "json", False))
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"token-forensics: {exc}", file=sys.stderr)
        return 2
