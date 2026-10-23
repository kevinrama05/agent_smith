# Agent Smith

An autonomous coding agent that solves **MBPP** and **SWE-bench** tasks using a
**Thought → Code → Observation** loop, with code executed in a restricted Python
sandbox that exposes tools over the **Model Context Protocol (MCP)**.

| Benchmark | Task | Agent output |
|---|---|---|
| **MBPP** | Write a small Python function that passes given `assert` tests | The function's source code |
| **SWE-bench** | Fix a real bug in a real repository (inside a Docker container) | A `git diff` patch |

## How it works

1. The LLM produces a *Thought* (reasoning) and a *Code* block (Python).
2. The code is extracted from the response (Markdown fences, XML, JSON or ReAct styles are supported).
3. The code runs in a sandbox with an import allowlist, a filesystem allowlist,
   and time/memory limits. MCP tools (`run_tests`, `read_file`, `edit_file`, …)
   are exposed to it as plain Python functions.
4. The output (*Observation*) goes back to the LLM. The loop repeats until the
   LLM calls `final_answer(...)` or an iteration, token or time budget runs out.
5. Every step (tokens, timing, raw LLM output, sandbox I/O) is recorded in a
   `solution.json` file.

```
agent_mbpp.py / agent_swebench.py      CLI entry points
        │
agent_cli_common.py                    argument parsing, provider setup, output writing
        │
agent_loop.py ── AgentLoop.run()
   ├── llm_provider.py                 OpenAI-compatible HTTP client, key rotation, fallback
   ├── code_extraction.py              pulls Python code out of LLM text
   └── sandbox.py                      restricted exec + MCP tool connector (stdio / HTTP)
           ├── mcp_tools_mbpp.py       run_tests
           └── mcp_tools_swebench.py   read/edit/list/search files, run_tests, run_command, get_patch
```

The loop itself is benchmark-agnostic: only the task prompt and the attached MCP
server differ. The system prompt's tool list is generated from whichever MCP
server is connected.

## Requirements

- Python **3.10** (`>=3.10,<3.11`)
- [uv](https://docs.astral.sh/uv/)
- A POSIX OS (Linux/macOS), since the sandbox uses `SIGALRM` and `RLIMIT_AS`
- `git`, plus `docker` for real SWE-bench runs
- An API key for any OpenAI-compatible LLM provider (Groq, OpenRouter, Mistral, Gemini, …)

## Setup

```bash
git clone <this-repo-url> agent_smith
cd agent_smith
uv sync
```

Create a `.env` file in the repository root with your API keys. It is loaded
automatically and is ignored by git. Never put keys in source code.

```dotenv
LLM_API_KEY=...            # default key when --api-key-env is not given
GROQ_API_KEY_1=...         # any variable names work; pass them with --api-key-env
OPENROUTER_API_KEY_1=...
MISTRAL_API_KEY_1=...
```

## Usage

### MBPP

```bash
uv run python -m agent_mbpp \
  --task-file cache/mbpp_task.json \
  --output cache/mbpp_solution.json \
  --model-name "llama-3.3-70b-versatile" \
  --provider-url "https://api.groq.com/openai/v1" \
  --api-key-env GROQ_API_KEY_1 --api-key-env GROQ_API_KEY_2
```

| Flag | Default | Description |
|---|---|---|
| `--task-file PATH` | required | Input task JSON |
| `--output PATH` | required | Where to write `solution.json` |
| `--model-name NAME` | required | Model ID sent to the API |
| `--provider-url URL` | required | Base URL of an OpenAI-compatible API |
| `--api-key-env VAR` | `LLM_API_KEY` | Name of an env var holding a key; repeat it to rotate between keys |
| `--max-iterations N` | `10` | Maximum loop steps |
| `--sandbox-config PATH` | built-in | Sandbox config JSON (see `sandbox_template.json`) |
| `--max-input-tokens N` | `6000` | Cumulative input-token budget |
| `--max-output-tokens N` | `1500` | Cumulative output-token budget |
| `--max-output-tokens-per-step N` | `512` | `max_tokens` per request |
| `--timeout-seconds N` | `120` | Wall-clock limit for the run |

The exit code is `0` on success and `1` otherwise. A `solution.json` is always
written, even if the run fails.

### SWE-bench

```bash
# With Docker
uv run python -m agent_swebench \
  --task-file cache/swebench_task.json \
  --output cache/swebench_solution.json \
  --model-name "codestral-latest" \
  --provider-url "https://api.mistral.ai/v1" \
  --api-key-env MISTRAL_API_KEY_1

# Local development without Docker, against any git repo on disk
uv run python -m agent_swebench ... --testbed-path path/to/repo
```

SWE-bench uses the same common flags, with `--max-iterations` defaulting to `30`.
Its budgets are fixed: 300k input tokens, 10k output tokens, and a 900 s timeout.

### Interactive sandbox

```bash
uv run sandbox                                          # default config, no tools
uv run sandbox sandbox_template.json                    # custom config
uv run sandbox --mcp-stdio "python mcp_tools_mbpp.py"   # attach an MCP server over stdio
uv run sandbox --mcp-server http://localhost:8000/mcp   # attach an MCP server over HTTP
```

### MCP servers on their own

```bash
python mcp_tools_mbpp.py [--transport stdio|streamable-http] [--port 8000]
TESTBED_PATH=/path/to/repo python mcp_tools_swebench.py [--transport stdio|streamable-http] [--port 8000]
```

## Testing

```bash
uv run python test_mbpp.py             # MBPP run_tests checks
uv run python test_swebench.py         # SWE-bench tool tests
uv run python test_swebench.py --mcp   # also run them through a real MCP stdio client
```

`tools/mock_moulinette.py` is a local stand-in grader with built-in fixture
tasks. Use it to run the whole dump → run agent → validate cycle:

```bash
python tools/mock_moulinette.py dump mbpp --output cache/mbpp_task.json
uv run python -m agent_mbpp --task-file cache/mbpp_task.json --output cache/mbpp_solution.json ...
python tools/mock_moulinette.py validate mbpp cache/mbpp_task.json cache/mbpp_solution.json
```

`swebench-loop.sh` runs a batch of SWE-bench tasks repeatedly through the grader
and handles rate limits. It expects the grader project in `../moulinette`.

## Configuration files

- **`sandbox_template.json`**: import allowlist, allowed directories, per-step timeout (30 s) and memory limit (512 MB).
- **`provider_template.json`**: an example multi-provider fallback setup with priority order, several keys per provider, and cooldowns for rate-limited keys.

## Data formats

MBPP task:

```json
{
  "task_id": 129,
  "task_definition": "Write a function to calculate magic square.",
  "function_definition": "def magic_square_test(my_matrix):",
  "test_imports": [],
  "test_list": ["assert magic_square_test([[2,7,6],[9,5,1],[4,3,8]])==True"]
}
```

SWE-bench task:

```json
{
  "instance_id": "owner__repo-1234",
  "problem_statement": "Issue text...",
  "docker_image": "image/name:tag",
  "eval_script": "#!/bin/bash\n...",
  "hints_text": "",
  "repo": "owner/repo"
}
```

The output `solution.json` contains `success`, `solution`, the token and time
totals, the system prompt, and a `steps` array recording every iteration.

## Known limitations

- The sandbox blocks dangerous builtins and restricts imports, but it is not a true capability sandbox. Determined introspection (e.g. `__subclasses__()`) could escape it.
- The timeout and memory limits are POSIX-only and apply to the whole process while a step runs.
- MCP tool calls run outside the sandbox's restrictions. Only their own timeouts limit them.
- `find_references` is a plain text search, not semantic analysis.

See [CODEBASE.md](CODEBASE.md) for a detailed file-by-file reference.
