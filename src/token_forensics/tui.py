from __future__ import annotations

import curses
import io
import re
import textwrap
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from .cli import Inference, Session, correlate_external_activity, human_bytes, iter_rollouts, parse_session, render_session, session_preview, timeline_header
from .stats import cumulative_statistics, statistics_lines


class Theme:
    NORMAL = 0
    DIM = 1
    HEADER = 2
    SELECTED = 3
    GREEN = 4
    YELLOW = 5
    RED = 6
    CYAN = 7
    MAGENTA = 8


_colors_ready = False


def init_colors() -> None:
    global _colors_ready
    if not curses.has_colors():
        return
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(Theme.DIM, curses.COLOR_WHITE, -1)
        curses.init_pair(Theme.HEADER, curses.COLOR_CYAN, -1)
        curses.init_pair(Theme.SELECTED, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(Theme.GREEN, curses.COLOR_GREEN, -1)
        curses.init_pair(Theme.YELLOW, curses.COLOR_YELLOW, -1)
        curses.init_pair(Theme.RED, curses.COLOR_RED, -1)
        curses.init_pair(Theme.CYAN, curses.COLOR_CYAN, -1)
        curses.init_pair(Theme.MAGENTA, curses.COLOR_MAGENTA, -1)
        _colors_ready = True
    except curses.error:
        _colors_ready = False


def attr(theme: int, bold: bool = False) -> int:
    value = curses.color_pair(theme) if _colors_ready else curses.A_NORMAL
    if theme == Theme.DIM:
        value |= curses.A_DIM
    if bold:
        value |= curses.A_BOLD
    return value


def put(window: Any, y: int, x: int, text: str, style: int = 0) -> None:
    height, width = window.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    try:
        window.addnstr(y, x, text, max(0, width - x), style)
    except curses.error:
        pass


def set_cursor(visibility: int) -> None:
    try:
        curses.curs_set(visibility)
    except curses.error:
        pass


def load_previews(home: Path) -> list[dict[str, Any]]:
    paths = sorted(iter_rollouts(home), key=lambda path: path.stat().st_mtime, reverse=True)
    return [session_preview(path) for path in paths]


def matching(items: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    if not query:
        return items
    needle = query.casefold()
    return [item for item in items if needle in " ".join(str(value) for value in item.values()).casefold()]


def row_text(item: dict[str, Any], width: int) -> str:
    started = str(item.get("started", "")).replace("T", " ").replace("Z", "")[:19]
    session_id = str(item.get("id", ""))
    cwd = str(item.get("cwd", ""))
    if width >= 110:
        return f" {started:19}  {session_id:36}  {cwd}"
    if width >= 72:
        return f" {started[5:]:14}  {session_id[:13]:13}  {cwd}"
    return f" {session_id[:13]:13}  {Path(cwd).name or cwd}"


def line_style(line: str) -> int:
    lowered = line.lower()
    if "high uncached" in lowered or "failed" in lowered or "rate jump" in lowered:
        return attr(Theme.RED)
    if "uncached input" in lowered or "large tool" in lowered or "reasoning" in lowered:
        return attr(Theme.YELLOW)
    if "context_compacted" in lowered or "turn_aborted" in lowered or "thread_rolled_back" in lowered:
        return attr(Theme.MAGENTA)
    if "next tool" in lowered or "input from" in lowered or "by requested tool" in lowered:
        return attr(Theme.CYAN)
    if line.startswith(("SESSION", "TOTALS", "TOP SUSPECTS")):
        return attr(Theme.HEADER, bold=True)
    if line.startswith(("TIME ", "CALLS ")) or line.startswith("model ") or line.startswith("rollout "):
        return attr(Theme.DIM)
    return attr(Theme.NORMAL)


def session_lines(session: Session) -> list[str]:
    output = io.StringIO()
    with redirect_stdout(output):
        render_session(session, color=False)
    # The TUI paints this row separately so it stays fixed while the body scrolls.
    return [line for line in output.getvalue().splitlines() if line != timeline_header()]


INFERENCE_ROW = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+(\d+)\s")


def inference_details(step: Inference) -> list[str]:
    usage = step.usage
    cache_rate = usage.cached / usage.input * 100 if usage.input else 0
    context_rate = usage.input / step.context_window * 100 if step.context_window else 0
    rates = []
    gap_note = ""
    if step.rate_gap_seconds is not None and not step.rate_delta_attributable:
        minutes = int(step.rate_gap_seconds // 60)
        seconds = int(step.rate_gap_seconds % 60)
        gap_note = f" over {minutes}m {seconds}s; not attributable to this inference"
    if step.rate_primary is not None:
        rates.append(f"primary {step.rate_primary:g}%" + (f" (Δ +{step.primary_delta:g}{gap_note})" if step.primary_delta else ""))
    if step.rate_secondary is not None:
        rates.append(f"secondary {step.rate_secondary:g}%" + (f" (Δ +{step.secondary_delta:g}{gap_note})" if step.secondary_delta else ""))
    details = [
        f"    ├─ usage     input {usage.input:,}  cached {usage.cached:,} ({cache_rate:.1f}%)  uncached {usage.uncached:,}",
        f"    ├─ generated output {usage.output:,}  reasoning {usage.reasoning:,}  total {usage.total:,}",
        f"    ├─ context   {usage.input:,} / {step.context_window:,} ({context_rate:.1f}%)" if step.context_window else "    ├─ context   window unknown",
    ]
    limit_lines = textwrap.wrap("  ".join(rates) if rates else "not recorded", width=100, break_long_words=False, break_on_hyphens=False)
    details.append(f"    ├─ limits    {limit_lines[0]}")
    details.extend(f"    │            {line}" for line in limit_lines[1:])
    if step.external_inferences:
        details.append(
            f"    ├─ concurrent {step.external_inferences} local inferences in {step.external_sessions} other session(s), "
            f"{step.external_input_tokens:,} submitted input tokens"
        )
        for activity in step.external_activity:
            label = activity["session_id"]
            details.append(
                f"    ├─ session    {label}  {activity['inferences']} inferences  "
                f"{activity['input_tokens']:,} input tokens"
            )
            if activity.get("cwd"):
                details.append(f"    │            {activity['cwd']}")
    if step.outputs:
        details.append("    ├─ outputs   " + "  ".join(f"{name} ×{count}" for name, count in step.outputs.items()))
    for tool in step.tools:
        facts = [f"call {tool.call_id}"]
        if tool.duration_ms is not None:
            facts.append(f"{tool.duration_ms}ms")
        if tool.result_bytes:
            facts.append(f"result {human_bytes(tool.result_bytes)}")
        if tool.success is not None:
            facts.append("success" if tool.success else "FAILED")
        title = f" ({tool.title})" if tool.title else ""
        details.append(f"    ├─ tool      {tool.name}{title}  " + "  ".join(facts))
        if tool.action:
            action_lines = textwrap.wrap(tool.action, width=100, break_long_words=False, break_on_hyphens=False)
            details.append(f"    ├─ action    {action_lines[0]}")
            details.extend(f"    │            {line}" for line in action_lines[1:])
        if tool.arguments is not None:
            argument_text = json_compact(tool.arguments)
            argument_lines = textwrap.wrap(argument_text, width=100, break_long_words=False, break_on_hyphens=False) or [argument_text]
            details.append(f"    ├─ arguments {argument_lines[0]}")
            details.extend(f"    │            {line}" for line in argument_lines[1:])
        if tool.command:
            command = " ".join(tool.command.split())
            wrapped = textwrap.wrap(command, width=100, break_long_words=False, break_on_hyphens=False) or [command]
            details.append(f"    ├─ command   {wrapped[0]}")
            details.extend(f"    │            {line}" for line in wrapped[1:])
        if tool.workdir:
            details.append(f"    ├─ workdir   {tool.workdir}")
        if tool.result_bytes:
            details.append("    ├─ causality result enters the following inference, not this inference's input")
    if step.prior_tools:
        prior = []
        for tool in step.prior_tools:
            result = f" result {human_bytes(tool.result_bytes)}" if tool.result_bytes else ""
            prior.append(f"{tool.name}{result} [{tool.call_id}]")
        details.append("    ├─ input from " + ", ".join(prior))
    if step.warnings:
        details.append("    └─ signals   " + ", ".join(step.warnings))
    else:
        details.append("    └─ signals   none")
    return details


def json_compact(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def expanded_rows(lines: list[str], session: Session, expanded: set[int]) -> list[tuple[str, int | None, bool]]:
    rows: list[tuple[str, int | None, bool]] = []
    by_index = {step.index: step for step in session.inferences}
    current_inference: int | None = None
    for line in lines:
        match = INFERENCE_ROW.match(line)
        if match:
            current_inference = int(match.group(1))
        elif line and not line.startswith(" "):
            current_inference = None
        primary = match is not None
        rows.append((line, current_inference, primary))
        if primary and current_inference in expanded:
            rows.extend((detail, current_inference, False) for detail in inference_details(by_index[current_inference]))
    return rows


def row_group(rows: list[tuple[str, int | None, bool]], index: int) -> tuple[str, int]:
    inference_index = rows[index][1]
    return ("inference", inference_index) if inference_index is not None else ("line", index)


def move_group(rows: list[tuple[str, int | None, bool]], selected: int, direction: int) -> int:
    if not rows:
        return 0
    selected = min(max(0, selected), len(rows) - 1)
    current = row_group(rows, selected)
    if direction > 0:
        index = selected + 1
        while index < len(rows) and row_group(rows, index) == current:
            index += 1
        return min(index, len(rows) - 1)
    index = selected - 1
    if index < 0:
        return 0
    previous = row_group(rows, index)
    while index > 0 and row_group(rows, index - 1) == previous:
        index -= 1
    return index


def group_start(rows: list[tuple[str, int | None, bool]], index: int) -> int:
    if not rows:
        return 0
    index = min(max(0, index), len(rows) - 1)
    group = row_group(rows, index)
    while index > 0 and row_group(rows, index - 1) == group:
        index -= 1
    return index


def detail_view(screen: Any, item: dict[str, Any], home: Path) -> None:
    screen.erase()
    put(screen, 0, 0, " Loading session...", attr(Theme.HEADER, bold=True))
    screen.refresh()
    try:
        session = parse_session(Path(item["path"]))
        correlate_external_activity(session, home)
        lines = session_lines(session)
    except Exception as exc:  # Keep terminal state intact for malformed legacy rollouts.
        session = Session(path=str(item["path"]))
        lines = ["ANALYSIS ERROR", "", str(exc)]
    offset = 0
    selected = 0
    expanded: set[int] = set()
    while True:
        rows = expanded_rows(lines, session, expanded)
        selected = min(max(0, selected), max(0, len(rows) - 1))
        screen.erase()
        height, width = screen.getmaxyx()
        put(screen, 0, 0, " TOKEN FORENSICS  /  SESSION ANALYSIS", attr(Theme.HEADER, bold=True))
        put(screen, 0, max(0, width - 62), "↑↓ select  Enter/Space expand  u/d half-page  g/G  Esc back ", attr(Theme.DIM))
        put(screen, 1, 0, timeline_header(), attr(Theme.DIM, bold=True))
        visible = max(1, height - 3)
        if selected < offset:
            offset = selected
        if selected >= offset + visible:
            offset = selected - visible + 1
        offset = min(max(0, offset), max(0, len(rows) - visible))
        selected_group = row_group(rows, selected) if rows else None
        for index, (line, _, primary) in enumerate(rows[offset : offset + visible], 2):
            absolute = offset + index - 2
            style = line_style(line)
            is_selected = selected_group is not None and row_group(rows, absolute) == selected_group
            if is_selected:
                style |= curses.A_REVERSE
            if primary:
                style |= curses.A_BOLD
            put(screen, index, 0, line.ljust(max(0, width)) if is_selected else line, style)
        status = f" {selected + 1} / {len(rows)} rows  {len(expanded)} expanded "
        put(screen, height - 1, max(0, width - len(status) - 1), status, attr(Theme.DIM))
        screen.refresh()
        key = screen.getch()
        if key in (27, ord("q"), curses.KEY_LEFT, curses.KEY_BACKSPACE, 127):
            return
        if key in (curses.KEY_DOWN, ord("j")):
            selected = move_group(rows, selected, 1)
        elif key in (curses.KEY_UP, ord("k")):
            selected = move_group(rows, selected, -1)
        elif key == curses.KEY_NPAGE:
            selected = group_start(rows, selected + visible)
        elif key == curses.KEY_PPAGE:
            selected = group_start(rows, selected - visible)
        elif key == ord("d"):
            selected = group_start(rows, selected + max(1, visible // 2))
        elif key == ord("u"):
            selected = group_start(rows, selected - max(1, visible // 2))
        elif key == ord("g"):
            selected = 0
        elif key == ord("G"):
            selected = len(rows) - 1
        elif key in (10, 13, ord(" "), curses.KEY_RIGHT):
            inference_index = rows[selected][1] if rows else None
            if inference_index is not None:
                if inference_index in expanded:
                    expanded.remove(inference_index)
                    # Return selection to the inference row when collapsing a child detail.
                    selected = next((i for i, row in enumerate(expanded_rows(lines, session, expanded)) if row[1] == inference_index and row[2]), selected)
                else:
                    expanded.add(inference_index)
        elif key == curses.KEY_LEFT:
            inference_index = rows[selected][1] if rows else None
            if inference_index in expanded:
                expanded.remove(inference_index)
        elif key == curses.KEY_RESIZE:
            continue


def statistics_view(screen: Any, home: Path) -> None:
    screen.erase()
    put(screen, 0, 0, " Loading cumulative statistics...", attr(Theme.HEADER, bold=True))
    screen.refresh()
    try:
        lines = statistics_lines(cumulative_statistics(home))
    except Exception as exc:
        lines = ["STATISTICS ERROR", "", str(exc)]
    offset = 0
    while True:
        screen.erase()
        height, width = screen.getmaxyx()
        put(screen, 0, 0, " TOKEN FORENSICS  /  CUMULATIVE STATISTICS", attr(Theme.HEADER, bold=True))
        put(screen, 0, max(0, width - 43), "u/d half-page  PgUp/PgDn  g/G  Esc back ", attr(Theme.DIM))
        visible = max(1, height - 2)
        offset = min(max(0, offset), max(0, len(lines) - visible))
        for index, line in enumerate(lines[offset : offset + visible], 1):
            put(screen, index, 0, line, line_style(line))
        status = f" {offset + 1}-{min(len(lines), offset + visible)} / {len(lines)} lines "
        put(screen, height - 1, max(0, width - len(status) - 1), status, attr(Theme.DIM))
        screen.refresh()
        key = screen.getch()
        if key in (27, ord("q"), curses.KEY_LEFT, curses.KEY_BACKSPACE, 127):
            return
        if key in (curses.KEY_DOWN, ord("j")):
            offset += 1
        elif key in (curses.KEY_UP, ord("k")):
            offset -= 1
        elif key == ord("d"):
            offset += max(1, visible // 2)
        elif key == ord("u"):
            offset -= max(1, visible // 2)
        elif key in (curses.KEY_NPAGE, ord(" ")):
            offset += visible
        elif key == curses.KEY_PPAGE:
            offset -= visible
        elif key == ord("g"):
            offset = 0
        elif key == ord("G"):
            offset = len(lines)


def browser(screen: Any, home: Path) -> None:
    set_cursor(0)
    screen.keypad(True)
    init_colors()
    items = load_previews(home)
    query = ""
    selected = 0
    top = 0
    filtering = False

    while True:
        filtered = matching(items, query)
        selected = min(max(0, selected), max(0, len(filtered) - 1))
        screen.erase()
        height, width = screen.getmaxyx()
        put(screen, 0, 0, " TOKEN FORENSICS", attr(Theme.HEADER, bold=True))
        put(screen, 0, 18, f"{len(filtered)} sessions", attr(Theme.DIM))
        help_text = "↑↓ navigate  Enter analyze  s statistics  / filter  r refresh  q quit "
        put(screen, 0, max(0, width - len(help_text) - 1), help_text, attr(Theme.DIM))
        if width >= 110:
            put(screen, 2, 0, " STARTED              SESSION ID                            WORKING DIRECTORY", attr(Theme.DIM))
        else:
            put(screen, 2, 0, " RECENT SESSIONS", attr(Theme.DIM))

        visible = max(1, height - 5)
        if selected < top:
            top = selected
        if selected >= top + visible:
            top = selected - visible + 1
        for row, item in enumerate(filtered[top : top + visible], 3):
            absolute = top + row - 3
            is_selected = absolute == selected
            style = attr(Theme.SELECTED, bold=True) if is_selected else attr(Theme.NORMAL)
            text = row_text(item, width)
            put(screen, row, 0, text.ljust(width) if is_selected else text, style)

        prompt = f" / {query}" if filtering or query else " Enter analyzes the selected session"
        put(screen, height - 1, 0, prompt, attr(Theme.CYAN if filtering else Theme.DIM))
        if filtering:
            set_cursor(1)
            try:
                screen.move(height - 1, min(width - 2, len(query) + 3))
            except curses.error:
                pass
        else:
            set_cursor(0)
        screen.refresh()
        key = screen.getch()

        if filtering:
            if key in (10, 13, 27):
                filtering = False
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                query = query[:-1]
                selected = top = 0
            elif 32 <= key <= 126:
                query += chr(key)
                selected = top = 0
            continue

        if key in (ord("q"), 27):
            return
        if key in (curses.KEY_DOWN, ord("j")):
            selected += 1
        elif key in (curses.KEY_UP, ord("k")):
            selected -= 1
        elif key == curses.KEY_NPAGE:
            selected += visible
        elif key == curses.KEY_PPAGE:
            selected -= visible
        elif key == ord("g"):
            selected = 0
        elif key == ord("G"):
            selected = len(filtered) - 1
        elif key == ord("/"):
            filtering = True
        elif key == ord("r"):
            items = load_previews(home)
            selected = top = 0
        elif key == ord("s"):
            statistics_view(screen, home)
        elif key in (10, 13, curses.KEY_RIGHT):
            if filtered:
                detail_view(screen, filtered[selected], home)
        elif key == curses.KEY_RESIZE:
            continue


def run_tui(home: Path) -> None:
    curses.wrapper(browser, home)
