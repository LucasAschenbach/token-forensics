# token-forensics

[![CI](https://github.com/LucasAschenbach/token-forensics/actions/workflows/ci.yml/badge.svg)](https://github.com/LucasAschenbach/token-forensics/actions/workflows/ci.yml)

Inspect where Codex token usage comes from, one model inference and tool loop at a time.

`token-forensics` is a local, read-only terminal application for the rollout data stored in `~/.codex`. It reconstructs session timelines, connects tool results to the model requests that consumed them, correlates account-wide rate-limit snapshots across local sessions, and summarizes usage by model, session, tool, and subagent.

> [!IMPORTANT]
> This is an independent analysis tool, not an official OpenAI product. Codex's local storage format is not a stable public API and may change between releases.

## Install

Install from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/LucasAschenbach/token-forensics.git
```

Or install a local checkout for development:

```bash
git clone https://github.com/LucasAschenbach/token-forensics.git
cd token-forensics
uv tool install --editable .
```

Python 3.10 or newer is required. The interactive UI targets macOS and Linux terminals.

## Use

Open the interactive session browser:

```bash
token-forensics
```

Home screen controls:

| Key | Action |
| --- | --- |
| `↑` / `↓`, `j` / `k` | Select a session |
| `Enter` | Analyze the selected session |
| `/` | Filter sessions |
| `s` | Open cumulative statistics |
| `r` | Refresh |
| `q` | Quit |

Analysis controls:

| Key | Action |
| --- | --- |
| `Enter` / `Space` | Expand or collapse an inference block |
| `u` / `d` | Move half a page |
| `Page Up` / `Page Down` | Move a full page |
| `g` / `G` | Jump to the top or bottom |
| `Esc` | Return |

The expanded view includes exact token counters, cache reuse, context utilization, rate-limit snapshots, normalized tool names, arguments, commands, working directories, result sizes, and causal predecessor tools.

## Commands

```bash
# List recent sessions
token-forensics list
token-forensics list my-project --limit 10

# Analyze by unique session-ID fragment or rollout path
token-forensics analyze 019f4d78
token-forensics analyze ~/.codex/sessions/2026/07/10/rollout-....jsonl

# Show cumulative model, session, tool, and subagent statistics
token-forensics stats

# Emit structured data
token-forensics analyze 019f4d78 --json
token-forensics stats --json

# Print the installed version
token-forensics --version

# Preserve color through a pager
token-forensics --color always analyze 019f4d78 | less -R
```

Set `CODEX_HOME` or pass `--codex-home` when Codex state lives somewhere other than `~/.codex`:

```bash
token-forensics --codex-home /path/to/codex-state stats
```

## What It Shows

Each inference row reports the usage in Codex's `last_token_usage` event:

- input and cached-input tokens;
- derived uncached input;
- output and reasoning-output tokens;
- context-window utilization; and
- primary and secondary rate-limit snapshots.

Tool rows preserve causality:

```text
model inference
  └─ next tool → xcodebuildmcp.snapshot_ui  result 4.1KiB

next model inference
  ↳ input from ← xcodebuildmcp.snapshot_ui result 4.1KiB
```

The tool execution itself does not receive a fictional token charge. Tool arguments are model output from the requesting inference; the tool result contributes to the following inference's input.

Code-mode wrappers are normalized into useful labels such as:

- `shell.exec`
- `tool_catalog.search`
- `tool_schema.inspect`
- `xcodebuildmcp.build_run_sim`
- `xcodebuildmcp.snapshot_ui`
- `agent.spawn`

The cumulative statistics page reports:

- per-model sessions, inference counts, average input, cache rate, and token totals;
- sessions with the most submitted input;
- normalized tool-call counts; and
- parent/child subagent sessions and child usage.

## Signals And Attribution

Color is reserved for actionable signals:

- unusually high uncached input;
- large tool results;
- heavy reasoning output;
- tool failures;
- cold-cache requests; and
- observed rate-limit changes.

Context utilization remains visible but is not treated as an anomaly by itself.

Rate-limit percentages are account-wide snapshots, not per-request charges. When snapshots in one session are separated by more than five minutes, the analyzer scans other local rollouts and lists concurrent sessions, inference counts, and submitted input. A percentage change with insufficient local evidence is marked as uncertain rather than assigned to a single request.

Tool-category totals describe inference usage associated with requests for that tool. Categories may overlap when one inference requests several tools and therefore do not necessarily sum to the session total.

## Privacy

All analysis runs locally. The tool:

- reads rollout JSONL and selected Codex SQLite state;
- does not read `auth.json`;
- does not modify `~/.codex`;
- makes no network requests; and
- does not send telemetry.

Rollouts may contain prompts, commands, file paths, and tool results. Treat JSON exports as sensitive and review them before sharing.

## Development

Run directly from a checkout:

```bash
PYTHONPATH=src python3 -m token_forensics
```

Run the test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Build distributions:

```bash
uv build
```

## License

MIT. See [LICENSE](LICENSE).
