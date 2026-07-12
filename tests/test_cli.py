import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from token_forensics.cli import classify_tool, main, parse_session, parse_tool_input, resolve_session, session_to_dict, summarize_command, timeline_header
from token_forensics.tui import expanded_rows, group_start, inference_details, move_group, row_group, session_lines


def write_rollout(path: Path) -> None:
    events = [
        {"timestamp": "2026-01-01T00:00:00Z", "type": "session_meta", "payload": {"id": "session-123", "cwd": "/tmp/example"}},
        {"timestamp": "2026-01-01T00:00:01Z", "type": "turn_context", "payload": {"model": "gpt-test", "effort": "high"}},
        {"timestamp": "2026-01-01T00:00:02Z", "type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "call_id": "call-1"}},
        {"timestamp": "2026-01-01T00:00:03Z", "type": "event_msg", "payload": {"type": "exec_command_end", "call_id": "call-1", "success": True, "duration": {"secs": 1, "nanos": 0}, "stdout": "ok"}},
        {"timestamp": "2026-01-01T00:00:03Z", "type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": {"input_tokens": 80000, "cached_input_tokens": 70000, "output_tokens": 500, "reasoning_output_tokens": 100, "total_tokens": 80500}, "model_context_window": 100000}, "rate_limits": {"primary": {"used_percent": 10}, "secondary": {"used_percent": 20}}}},
        {"timestamp": "2026-01-01T00:00:04Z", "type": "response_item", "payload": {"type": "message"}},
        {"timestamp": "2026-01-01T00:00:05Z", "type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": {"input_tokens": 90000, "cached_input_tokens": 70000, "output_tokens": 100, "reasoning_output_tokens": 0, "total_tokens": 90100}, "model_context_window": 100000}, "rate_limits": {"primary": {"used_percent": 13}, "secondary": {"used_percent": 20}}}},
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


class CliTests(unittest.TestCase):
    def test_parse_session_relates_tool_to_next_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout-session-123.jsonl"
            write_rollout(path)
            session = parse_session(path)

            self.assertEqual(session.session_id, "session-123")
            self.assertEqual(session.model, "gpt-test")
            self.assertEqual(len(session.inferences), 2)
            self.assertEqual(session.inferences[0].tools[0].name, "exec_command")
            self.assertEqual(session.inferences[1].prior_tools[0].call_id, "call-1")
            self.assertEqual(session.inferences[1].usage.uncached, 20000)
            self.assertIn("rate jump observed", session.inferences[1].warnings)
            json.dumps(session_to_dict(session))

    def test_resolve_unique_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions"
            sessions.mkdir()
            expected = sessions / "rollout-unique-abcdef.jsonl"
            expected.write_text("")
            self.assertEqual(resolve_session("abcdef", root), expected)

    def test_bare_command_defaults_to_session_list(self) -> None:
        output = StringIO()
        with patch("token_forensics.cli.codex_home", return_value=Path("/nonexistent")), redirect_stdout(output):
            self.assertEqual(main([]), 0)
        self.assertIn("SESSION", output.getvalue())

    def test_version_flag(self) -> None:
        output = StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), "token-forensics 0.1.0")

    def test_timeline_header_uses_data_column_widths(self) -> None:
        header = timeline_header()
        self.assertEqual(header.index("#"), 16)
        self.assertEqual(header.index("INPUT"), 20)
        self.assertEqual(header.index("CACHE"), 28)
        self.assertEqual(header.index("UNCACH"), 35)
        self.assertEqual(header.index("OUTPUT"), 43)
        self.assertEqual(header.index("REASON"), 51)
        self.assertEqual(header.index("CTX"), 61)
        self.assertEqual(header.index("RATE"), 65)

    def test_code_mode_exec_is_made_readable(self) -> None:
        payload = {
            "input": 'const r = await tools.exec_command({cmd:"sed -n \'1,20p\' \'/tmp/one/SKILL.md\'; sed -n \'1,20p\' \'/tmp/two/SKILL.md\'","workdir":"/tmp/project"});'
        }
        command, workdir = parse_tool_input(payload)
        self.assertEqual(workdir, "/tmp/project")
        self.assertIn("/tmp/one/SKILL.md", command or "")
        self.assertEqual(summarize_command(command), "read 2 files: one/SKILL.md, two/SKILL.md")

    def test_code_mode_xcode_tools_are_unwrapped(self) -> None:
        name, action = classify_tool(
            {"input": 'const r=await tools.mcp__xcodebuildmcp__build_run_sim({});'}, "exec"
        )
        self.assertEqual(name, "xcodebuildmcp.build_run_sim")
        self.assertIn("MCP", action or "")

        name, action = classify_tool(
            {"input": 'const t=ALL_TOOLS.find(x=>x.name==="mcp__xcodebuildmcp__list_sims");'}, "exec"
        )
        self.assertEqual(name, "tool_schema.inspect")
        self.assertIn("xcodebuildmcp.list_sims", action or "")

        name, _ = classify_tool({"input": 'ALL_TOOLS.filter(x => /XcodeBuildMCP/.test(x.name));'}, "exec")
        self.assertEqual(name, "tool_catalog.search")

    def test_subagent_spawn_is_normalized(self) -> None:
        name, action = classify_tool(
            {"arguments": '{"agent_type":"worker","model":"gpt-5.4-mini"}'}, "spawn_agent"
        )
        self.assertEqual(name, "agent.spawn")
        self.assertEqual(action, "spawn worker using gpt-5.4-mini")

    def test_inference_rows_expand_with_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout-session-123.jsonl"
            write_rollout(path)
            session = parse_session(path)
            rows = expanded_rows(session_lines(session), session, {1})
            details = inference_details(session.inferences[0])
            following_details = inference_details(session.inferences[1])

            self.assertTrue(any("usage" in line for line in details))
            self.assertTrue(any("call-1" in line for line in details))
            self.assertTrue(any("input from" in line and "result" in line and "call-1" in line for line in following_details))
            self.assertTrue(any(not primary and index == 1 for _, index, primary in rows))

            first = next(index for index, row in enumerate(rows) if row[1] == 1)
            second = move_group(rows, first, 1)
            self.assertEqual(row_group(rows, first), ("inference", 1))
            self.assertEqual(row_group(rows, second), ("inference", 2))
            self.assertTrue(all(row_group(rows, index) == ("inference", 1) for index in range(first, second)))
            self.assertEqual(group_start(rows, second + 1), second)


if __name__ == "__main__":
    unittest.main()
