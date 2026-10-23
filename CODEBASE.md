# Agent Smith — Codebase Guide

This document explains every file in the repository: what each part does, why it
exists, and every parameter, flag, config value, and environment variable you
need to run the program.

---

## Table of contents

1. [What the program does](#1-what-the-program-does)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [Requirements and setup](#3-requirements-and-setup)
4. [Running the program (all commands and parameters)](#4-running-the-program)
5. [File-by-file reference](#5-file-by-file-reference)
   - [pyproject.toml / .python-version / uv.lock](#pyprojecttoml--python-version--uvlock)
   - [.gitignore](#gitignore)
   - [.env](#env)
   - [provider_template.json](#provider_templatejson)
   - [sandbox_template.json](#sandbox_templatejson)
   - [src/agent_smith/\_\_init\_\_.py](#srcagent_smith__init__py)
   - [\_\_init\_\_.py (root)](#__init__py-root)
   - [agent_loop.py](#agent_looppy)
   - [llm_provider.py](#llm_providerpy)
   - [code_extraction.py](#code_extractionpy)
   - [sandbox.py](#sandboxpy)
   - [mcp_tools_mbpp.py](#mcp_tools_mbpppy)
   - [mcp_tools_swebench.py](#mcp_tools_swebenchpy)
   - [agent_cli_common.py](#agent_cli_commonpy)
   - [agent_mbpp.py](#agent_mbpppy)
   - [agent_swebench.py](#agent_swebenchpy)
   - [test_mbpp.py](#test_mbpppy)
   - [test_swebench.py](#test_swebenchpy)
   - [tools/mock_moulinette.py](#toolsmock_moulinettepy)
   - [cache/ and graphify-out/](#cache-and-graphify-out)
6. [Data formats (input task / output solution)](#6-data-formats)
7. [Limits and defaults cheat sheet](#7-limits-and-defaults-cheat-sheet)
8. [Known limitations](#8-known-limitations)

---

## 1. What the program does

Agent Smith is an **autonomous coding agent** that solves two kinds of benchmark tasks:

| Benchmark | Task | What the agent returns |
|---|---|---|
| **MBPP** | Write a small Python function that passes given `assert` tests | The function's source code |
| **SWE-bench** | Fix a real bug in a real repository (inside a Docker container) | A `git diff` patch |

It works in a **Thought → Code → Observation** loop:

1. An LLM is asked for a *Thought* (reasoning) and a *Code* block (Python).
2. The code is extracted from the LLM's text.
3. The code runs inside a restricted Python **sandbox**. The sandbox exposes
   **MCP tools** (e.g. `run_tests`, `read_file`, `edit_file`) as plain Python
   functions.
4. The output (*Observation*) is sent back to the LLM, and the loop repeats until
   the LLM calls `final_answer(...)` or a limit (iterations, tokens, time) is hit.
5. Everything (each step's tokens, timing, raw LLM text, sandbox I/O) is written
   to a `solution.json` file that a grader (the "moulinette") validates.

---

## 2. Architecture at a glance

```
             agent_mbpp.py / agent_swebench.py          (CLI entry points)
                           │ uses
                    agent_cli_common.py                  (arg parsing, provider build, output writing)
                           │
                           ▼
                     agent_loop.py  ── AgentLoop.run() ──────────────┐
                     │          │                                    │
         generate()  │          │ extract_code()                     │ execute()
                     ▼          ▼                                    ▼
             llm_provider.py   code_extraction.py              sandbox.py (Sandbox)
             (HTTP to LLM,     (```python``` / XML /           restricted exec + final_answer
              key rotation,     JSON / ReAct → Python)                │
              fallback)                                               │ MCPToolConnector (stdio/HTTP)
                                                                      ▼
                                              mcp_tools_mbpp.py      or     mcp_tools_swebench.py
                                              (run_tests)                   (read/edit/search/run_tests/get_patch...)
                                                                            runs on host (--testbed-path)
                                                                            or inside Docker via `docker exec`
```

Key design idea: **the loop doesn't know which benchmark it is running.** The
difference between MBPP and SWE-bench lives only in (a) the task prompt and
(b) which MCP server is attached to the sandbox. The system prompt's tool list is
generated automatically from whatever MCP server is connected.

---

## 3. Requirements and setup

| Requirement | Value | Where it's declared |
|---|---|---|
| Python | **3.10 exactly** (`>=3.10,<3.11`) | `pyproject.toml`, `.python-version` |
| Package manager | [`uv`](https://docs.astral.sh/uv/) | `uv.lock`, `uv_build` backend |
| `requests` | `>=2.32.0` — HTTP calls to LLM APIs | `pyproject.toml` |
| `mcp` | `>=2.2.0` — MCP client & server (**v2 API**) | `pyproject.toml` |
| `pydantic` | `>=2.13.5` — config/task/output models | `pyproject.toml` |
| `python-dotenv` | `>=1.2.3` — auto-load `.env` | `pyproject.toml` |
| `git` | needed by SWE-bench tools (`get_patch`) and tests | system |
| `docker` | needed for real SWE-bench runs (not for MBPP or `--testbed-path`) | system |
| POSIX OS (Linux/macOS) | for sandbox timeout (`SIGALRM`) and memory limit (`RLIMIT_AS`) | `sandbox.py` |

Setup:

```bash
uv sync                 # create .venv and install dependencies
# create a .env file in the repo root with your API keys — see §5 ".env"
```

---

## 4. Running the program

All commands are run from the repository root.

### 4.1 MBPP agent

```bash
uv run python -m agent_mbpp \
  --task-file cache/mbpp_task.json \
  --output cache/mbpp_solution.json \
  --model-name "llama-3.3-70b-versatile" \
  --provider-url "https://api.groq.com/openai/v1" \
  --api-key-env GROQ_API_KEY_1 --api-key-env GROQ_API_KEY_2
```

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--task-file PATH` | **yes** | — | JSON task file (see §6) |
| `--output PATH` | **yes** | — | Where to write `solution.json` (parent dirs are created) |
| `--model-name NAME` | **yes** | — | Model id sent to the API, e.g. `qwen/qwen3-235b-a22b-2507` |
| `--provider-url URL` | **yes** | — | Base URL of an OpenAI-compatible API (the code appends `/chat/completions`) |
| `--api-key-env VAR` | no (repeatable) | `LLM_API_KEY` | **Name** of an env var holding an API key. Repeat to enable key rotation |
| `--max-iterations N` | no | `10` | Max Thought/Code steps |
| `--sandbox-config PATH` | no | built-in defaults | `SandboxConfig` JSON (e.g. `sandbox_template.json`) |
| `--max-input-tokens N` | no | `6000` | Cumulative input-token budget for the whole run |
| `--max-output-tokens N` | no | `1500` | Cumulative output-token budget for the whole run |
| `--max-output-tokens-per-step N` | no | `512` | Per-request `max_tokens` (also clamped by remaining budget) |
| `--timeout-seconds N` | no | `120` | Wall-clock limit for the whole run |

Exit code: `0` if `success=true`, `1` otherwise. A `solution.json` is **always**
written, even on crashes (missing key, bad task file, etc.).

### 4.2 SWE-bench agent

```bash
# Real mode (Docker):
uv run python -m agent_swebench \
  --task-file cache/swebench_task.json \
  --output cache/swebench_solution.json \
  --model-name "qwen/qwen3-235b-a22b-2507" \
  --provider-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY_1

# Local dev mode (no Docker), against any git repo on disk:
uv run python -m agent_swebench ... --testbed-path cache/fixture_repo
```

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--task-file`, `--output`, `--model-name`, `--provider-url`, `--api-key-env`, `--sandbox-config` | same as MBPP | | |
| `--max-iterations N` | no | `30` | Max steps |
| `--testbed-path PATH` | no | — | **Local dev only.** Skip Docker; run the SWE-bench MCP server on the host against this repo |

Not overridable from the CLI (hard-coded in `agent_swebench.py`): input tokens
`300000`, output tokens `10000`, timeout `900 s`, per-step output `512`.

### 4.3 Interactive sandbox REPL

```bash
uv run sandbox                                              # defaults, no tools
uv run sandbox sandbox_template.json                        # custom config
uv run sandbox --mcp-stdio "python mcp_tools_mbpp.py"       # attach MCP server over stdio
uv run sandbox --mcp-server http://localhost:8000/mcp       # attach MCP server over HTTP
```

| Arg | Meaning |
|---|---|
| `config` (positional, optional) | Path to `SandboxConfig` JSON |
| `--mcp-stdio "CMD"` | Command line that launches an MCP server over stdio |
| `--mcp-server URL` | URL of an MCP server over streamable HTTP |

`--mcp-stdio` and `--mcp-server` are mutually exclusive. Type `exit` or Ctrl-D to quit.

### 4.4 MCP servers standalone

```bash
python mcp_tools_mbpp.py     [--transport stdio|streamable-http] [--port 8000]
TESTBED_PATH=/path/to/repo \
python mcp_tools_swebench.py [--transport stdio|streamable-http] [--port 8000]
```

### 4.5 Tests and local grading

```bash
uv run python test_mbpp.py              # unit checks for MBPP run_tests
uv run python test_swebench.py          # direct tests of SWE-bench tools
uv run python test_swebench.py --mcp    # + tests through a real MCP stdio client

python tools/mock_moulinette.py dump mbpp --output cache/mbpp_task.json [--task-id 1]
python tools/mock_moulinette.py validate mbpp cache/mbpp_task.json cache/mbpp_solution.json
python tools/mock_moulinette.py dump swebench --output cache/swebench_task.json
python tools/mock_moulinette.py validate swebench cache/swebench_task.json cache/swebench_solution.json
```

### 4.6 Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `LLM_API_KEY` | `agent_cli_common.build_provider` | Default API key when `--api-key-env` is not given |
| *any name you pass to `--api-key-env`* (e.g. `GROQ_API_KEY_1`) | `build_provider` | API key(s) for the provider |
| names in `provider_template.json` → `api_key_env_vars` | `llm_provider.load_provider_router` | Keys for multi-provider routing |
| `LLM_REASONING_EFFORT` | `llm_provider._default_reasoning_effort` | Override `reasoning_effort` sent to the API. Unset → `"low"` for reasoning models (gpt-oss, o1, o3, o4-mini), nothing for others. Set to empty string to disable entirely |
| `TESTBED_PATH` | `mcp_tools_swebench.py` | **Required** by the SWE-bench MCP server: root of the repo to work on |
| `EVAL_SCRIPT` | `mcp_tools_swebench.run_tests` | Optional: eval script contents. If unset, `$TESTBED_PATH/eval.sh` is used |

---

## 5. File-by-file reference

### `pyproject.toml` / `.python-version` / `uv.lock`

**Why:** project metadata and reproducible dependencies.

- `[project]` — name `agent-smith`, version `0.1.0`, Python `>=3.10,<3.11`, the four dependencies above.
- `[project.scripts]`
  - `agent-smith = "agent_smith:main"` → prints a usage hint.
  - `sandbox = "agent_smith:sandbox_main"` → launches the sandbox REPL (`uv run sandbox`).
- `[build-system]` — `uv_build` backend; it packages only `src/agent_smith/`.
- `.python-version` — pins `3.10` for `uv`/pyenv.
- `uv.lock` — exact resolved versions of every dependency (generated; don't edit by hand).

### `.gitignore`

Ignores Python caches, build output, `.venv`, **`.env` (secrets)**, `cache/`
(task dumps and solutions), `graphify-out/` (local tooling output), and
`en.subject.pdf` (the 42 subject, not part of the submission).

### `.env`

**Why:** keeps API keys out of the source code (hard-coded keys are a graded
security failure). Loaded automatically by `agent_cli_common.py` if present in
the current directory.

Expected variables (values are secrets — never commit):

```dotenv
LLM_API_KEY=...            # default key if --api-key-env is not passed
GROQ_API_KEY_1=...         # https://api.groq.com/openai/v1
GROQ_API_KEY_2=...
OPENROUTER_API_KEY_1=...   # https://openrouter.ai/api/v1
GEMINI_API_KEY=...         # https://generativelanguage.googleapis.com/v1beta/openai/
MISTRAL_API_KEY_1=...      # https://api.mistral.ai/v1
```

Only the variables you actually reference (via `--api-key-env` or a provider
config) need to be set.

### `provider_template.json`

**Why:** example config for **multi-provider fallback** (used by
`llm_provider.load_provider_router()`, e.g. for your own benchmarking — the graded
CLIs use a single provider built from flags instead).

Provider order = priority order. Each entry (`ProviderConfigEntry`):

| Field | Required | Default | Meaning |
|---|---|---|---|
| `name` | yes | — | Label used in errors/logs |
| `base_url` | yes | — | OpenAI-compatible base URL |
| `model` | yes | — | Model id to request from this provider |
| `api_key_env_vars` | yes | — | List of **env var names** (not keys!) |
| `extra_headers` | no | `{}` | Extra HTTP headers (OpenRouter's `HTTP-Referer`, `X-Title`) |
| `timeout_seconds` | no | `60` | Per-request HTTP timeout |
| `max_retries_per_key` | no | `2` | Attempts per key before rotating |
| `cooldown_seconds` | no | `60` | How long a rate-limited key is benched |

### `sandbox_template.json`

**Why:** example `SandboxConfig` you can pass with `--sandbox-config` or to `uv run sandbox`. Identical to the built-in defaults:

| Field | Default | Meaning |
|---|---|---|
| `authorized_imports` | `math`, `math.*`, `collections`, `collections.*`, `itertools`, `re`, `json`, `typing`, `typing.*`, `functools`, `operator`, `heapq`, `bisect`, `copy`, `string`, `random`, `datetime`, `datetime.*`, `array`, `cmath` | Import allowlist. `pkg.*` allows `pkg` and all submodules |
| `allowed_directories` | `/testbed`, `/tmp/agent` | Only paths inside these may be `open()`ed by sandboxed code |
| `max_execution_time_seconds` | `30` | Per-step execution timeout |
| `max_memory_mb` | `512` | Address-space limit during a step |

### `src/agent_smith/__init__.py`

**Why:** `uv_build` only installs the `src/agent_smith` package, but all real
code lives at the repo root (so `python -m agent_mbpp` and `from sandbox import ...`
work). This file bridges the two:

- `_ROOT` — repo root (two levels up); inserted into `sys.path` so root modules are importable from console scripts.
- `main()` — entry point for `uv run agent-smith`; prints the real CLI commands.
- `sandbox_main()` — entry point for `uv run sandbox`; imports and calls `sandbox.main()`.

### `__init__.py` (root)

Empty. Marks the root as a package; has no runtime behaviour.

---

### `agent_loop.py`

**Why:** the orchestrator — ties LLM, code extraction and sandbox together. One
`AgentLoop` runs one task and returns a `SolutionOutput`.

#### `StepMetrics` (pydantic model) — one entry per iteration

| Field | Meaning |
|---|---|
| `step` | 1-based iteration number |
| `input_tokens` / `output_tokens` | Tokens for this LLM call (from the API's `usage`) |
| `request_time_ms` | LLM call wall time |
| `timestamp` | ISO-8601 time recorded |
| `api_url` / `model_name` | Which endpoint/model served the step |
| `llm_output` | Raw LLM text before extraction |
| `sandbox_input` | Code actually executed (`""` if none found) |
| `sandbox_output` | Observation (stdout/stderr/error, plus any extraction note) |
| `retries` | API retries before success |

#### `SolutionOutput` (pydantic model) — what gets written to `solution.json`

`task_id`, `benchmark` (`"mbpp"`/`"swebench"`), `success`, `solution` (code or
patch), `iterations`, `total_requests` (includes retries),
`total_input_tokens`, `total_output_tokens`, `total_time_seconds`, `steps`,
`system_prompt` (full prompt, for provenance checking), `error`, `timestamp`.

#### `DEFAULT_EXAMPLE`

A worked SWE-bench-style example (search → read → edit + run_tests →
`final_answer(get_patch())`) inserted into the system prompt to teach the LLM the
format. MBPP overrides it with a shorter example to save tokens.

#### `build_system_prompt(sandbox_manual, task_hint="", example=DEFAULT_EXAMPLE)`

Builds the system prompt: loop rules (Thought/Code, `<end_code>` marker, don't
invent Observations, call `final_answer` once) + the **sandbox manual** (auto-generated
tool list from the connected MCP server) + the task prompt + the example.

#### `AgentLoop.__init__(...)` parameters

| Param | Default | Meaning |
|---|---|---|
| `provider` | — | Anything with `.generate(messages, stop, max_tokens, temperature) -> LLMResponse` (normally a `ProviderRouter`) |
| `sandbox` | — | A `Sandbox` with MCP tools attached |
| `max_iterations` | `10` | Step cap |
| `max_input_tokens` | `6000` | Cumulative input token budget |
| `max_output_tokens` | `1500` | Cumulative output token budget |
| `max_output_tokens_per_step` | `512` | Per-call `max_tokens` |
| `timeout_seconds` | `120` | Wall-clock budget |
| `stop_sequences` | `["<end_code>"]` | Where the LLM should stop generating |
| `example` | `DEFAULT_EXAMPLE` | Example block for the prompt |

#### `AgentLoop._estimate_next_input(steps, messages)`

Predicts the next request's input tokens (previous prompt + newly appended text
at ~3 chars/token + 32 overhead). Used to stop **before** a request would push the
cumulative input total over the limit (since every request resends the whole history).

#### `AgentLoop.run(task_prompt, task_id, benchmark)`

Per step:
1. Check timeout, predicted input budget, and output budget — stop with an `error` message if any is exceeded.
2. Call `provider.generate(...)` with `temperature=0.0` and `max_tokens=min(per_step, remaining_output_budget)`. Provider failures end the loop with an error.
3. `extract_code()` on the reply. If nothing executable is found, the explanation becomes the Observation (no sandbox call).
4. `sandbox.execute(code)`. If `final_answer(...)` was called → `success=True`, `solution=str(value)`, stop.
5. Otherwise append `Observation:\n...` as a user message and continue.

If the loop ends without `final_answer`, `error` explains why (max iterations, token limit, timeout, provider failure).

---

### `llm_provider.py`

**Why:** abstracts the LLM API so the agent loop never deals with HTTP, keys,
retries, or which provider served the request.

#### Exceptions

| Class | When |
|---|---|
| `LLMProviderError` | Base class; also used for non-retryable 4xx and bad response shapes |
| `RateLimitError(message, retry_after)` | HTTP 429; carries `Retry-After` seconds if present |
| `TransientProviderError` | Timeouts, connection errors, HTTP 5xx (retryable) |
| `AllKeysExhaustedError` | Every key of one provider is cooling down / failed |
| `AllProvidersExhaustedError` | Every provider in the router failed — surfaces to the agent loop |

#### `LLMResponse` (dataclass)

`text`, `input_tokens`, `output_tokens`, `request_time_ms`, `api_url`,
`model_name`, `retries`, `raw` (full JSON payload). Maps 1:1 onto `StepMetrics`.

#### `KeyRotator(keys, cooldown_seconds=60.0)`

Round-robins through API keys, skipping ones in cooldown.
- `get_key()` — next available key or `AllKeysExhaustedError`.
- `mark_rate_limited(key)` — bench a key for `cooldown_seconds`.
- `has_available_key()` — whether any key is usable now.

#### `LLMProvider`

Abstract interface: `generate(messages, model, stop=None, max_tokens=1500, temperature=0.0) -> LLMResponse`.

#### Reasoning-model helpers

- `_REASONING_MODEL_MARKERS = ("gpt-oss", "o1", "o3", "o4-mini")`, `_is_reasoning_model(model)`.
- `_default_reasoning_effort(model)` — `"low"` for reasoning models (so hidden reasoning doesn't eat the whole token budget), overridable via `LLM_REASONING_EFFORT`.
- `_truncate_at_stop(text, stop)` — applies stop sequences **client-side** for reasoning models, because server-side `stop` would also cut the hidden reasoning and return empty content.

#### `OpenAICompatibleProvider(...)`

Works with any `POST {base_url}/chat/completions` API (Groq, OpenRouter,
Together, Fireworks, Mistral, Gemini's OpenAI endpoint…).

| Param | Default | Meaning |
|---|---|---|
| `name` | — | Label |
| `base_url` | — | Base URL (trailing `/` stripped) |
| `api_keys` | — | List of actual key strings |
| `extra_headers` | `None` | Additional headers |
| `timeout_seconds` | `60.0` | HTTP timeout |
| `max_retries_per_key` | `2` | Attempts per key |
| `cooldown_seconds` | `60.0` | Rate-limit cooldown per key |
| `max_rate_limit_wait_seconds` | `30.0` | If no other key is free and `Retry-After` ≤ this, sleep and retry the same key instead of giving up |

`generate()` loops over keys; for each key retries on `RateLimitError` /
`TransientProviderError` with exponential backoff + jitter (`_backoff_delay`).
`_single_request()` builds the payload, sends it, maps HTTP codes to exceptions,
and if a provider rejects `reasoning_effort` (HTTP 400) it disables it and retries once.

#### `RoutedProvider(provider, model)` / `ProviderRouter(routed_providers)`

The router tries each `(provider, model)` pair in order; on
`AllKeysExhaustedError` **or** any `LLMProviderError` it falls back to the next.
Its `generate(messages, stop, max_tokens, temperature)` has no `model` argument —
the agent loop never chooses a model.

#### `ProviderConfigEntry`, `RouterConfig`, `load_provider_router(config_path)`

Loads a JSON file shaped like `provider_template.json`, resolves every
`api_key_env_vars` entry from the environment (raises `ValueError` at startup if
one is missing), and returns a ready `ProviderRouter`.

---

### `code_extraction.py`

**Why:** LLMs don't always follow the format. This layer turns many formats into
plain Python so the sandbox only ever sees Python, and it **always tells the LLM**
what went wrong or what was tolerated.

#### `ExtractionResult` (dataclass)

`code` (Python to run, `""` if none), `found` (bool), `source_format`
(`python_block` | `xml_invoke` | `json_tool_call` | `react` | `none`), `note`
(feedback text for the LLM).

#### Parsers (tried in this order; first match wins)

| Function | Recognises | Notes |
|---|---|---|
| `_extract_python_block` | ```` ```python ... ``` ```` | Also tolerates a bare ```` ``` ```` fence and a missing closing fence, with a note |
| `_extract_xml_invoke` | `<invoke name="tool"><parameter name="x">v</parameter></invoke>` | → `result = tool(x=...)\nprint(result)` |
| `_extract_json_tool_call` | `<tool_call>{"name":..., "arguments":{...}}</tool_call>` | Bad JSON / missing `name` → `found=False` with explanation |
| `_extract_react` | `Action: tool` / `Action Input: {...}` | Non-JSON input becomes a single string argument |

Helpers: `_strip_trailing_markers` removes leftover `<end_code>`, `</tool_call>`,
`</invoke>`; `_to_python_literal` turns raw strings into Python literals (JSON first,
then string).

#### Public functions

- `extract_code(llm_output)` — runs the parsers; empty or unrecognised output returns `found=False` with an instruction to use a ```` ```python ```` block.
- `feedback_text(result)` — text to prepend to (or replace) the Observation: the error message if nothing found, `[Note: ...]` if something was tolerated, else `""`.

---

### `sandbox.py`

**Why:** runs LLM-written code with restrictions, provides `final_answer`, and
bridges MCP tools into the code's namespace.

#### `SandboxConfig` (pydantic)

See [sandbox_template.json](#sandbox_templatejson) for fields and defaults.

#### `ExecutionResult` (dataclass)

`output` (the Observation text), `is_final_answer`, `final_answer_value`, `truncated`.
`MAX_OUTPUT_CHARS = 8000` — longer outputs are cut with a `[truncated ...]` note.

#### Internal signals

`_FinalAnswerSignal(value)` (raised by `final_answer`) and `_SandboxTimeout`
(raised by the alarm). These are control flow, not errors.

#### Restrictions

| Piece | What it does |
|---|---|
| `_import_is_authorized(name, authorized)` | Exact match or `pkg.*` wildcard match |
| `_make_restricted_import(authorized)` | Replacement `__import__` that raises `ImportError` listing allowed modules |
| `SandboxPermissionError` | Raised on disallowed file access |
| `_make_restricted_open(allowed_directories)` | Replacement `open()` that resolves symlinks/`..` then checks the path is inside an allowed dir |
| `_DANGEROUS_BUILTINS` | Removed: `eval`, `exec`, `compile`, `input`, `breakpoint`, `help`, `globals`, `locals`, `vars`, `dir`, `exit`, `quit`, `memoryview` |
| `_build_safe_builtins(...)` | All public builtins minus the dangerous ones, with restricted `__import__` and `open` |

#### `Sandbox(config)`

- `namespace` — persists across steps (variables from step N are visible in step N+1).
- `extra_namespace` — MCP tool wrappers merged in before each execution.
- `attach_mcp(connector)` — registers a connector's tools (call again to swap servers).
- `_inject_final_answer()` — always provides `final_answer(value)`.
- `_apply_limits()` / `_clear_limits()` — `SIGALRM` timeout and `RLIMIT_AS` memory cap (POSIX only; silently skipped elsewhere).
- `execute(code)` — compiles (syntax errors reported without running), runs with stdout/stderr captured, and converts every outcome into an `ExecutionResult`: final answer, timeout (with partial output), `MemoryError`, traceback for any other exception. `KeyboardInterrupt`/`SystemExit` **propagate** on purpose.
- `manual_text()` — the "sandbox manual" injected into the system prompt: `final_answer`, discovered MCP tools with signatures, allowed imports/dirs, limits.

#### `MCPToolConnector(mode, target, extra_env=None)`

| Param | Meaning |
|---|---|
| `mode` | `"stdio"` or `"http"` |
| `target` | stdio: command line to launch the server (split with `shlex`); http: server URL |
| `extra_env` | stdio only: env vars added on top of the current environment (e.g. `{"TESTBED_PATH": ...}`) |

The MCP client is async but sandbox code is sync, so the connector runs a private
asyncio event loop in a **background thread** and submits calls to it with
`run_coroutine_threadsafe`.

- `connect(timeout=30.0)` — start the loop, open the session, `initialize()`, `list_tools()` → `tools_info`. Raises `MCPConnectionError` on failure.
- `close()` — best-effort shutdown of session, loop and thread.
- `_call_tool_sync(name, kwargs, timeout=120.0)` — call a tool and flatten its content to text; tool errors become `"Error from tool '...': ..."`.
- `build_wrappers()` / `_make_wrapper(tool)` — one Python function per tool; positional args are mapped onto the schema's parameter order, so `read_file("a.py", 1, 50)` and `read_file(filepath="a.py")` both work.
- `manual_text()` — signature + description lines generated from each tool's JSON schema.

#### CLI: `_load_config(path)`, `_run_repl(sandbox)`, `main()`

Interactive REPL (see §4.3). Each line is executed in the same persistent sandbox.

---

### `mcp_tools_mbpp.py`

**Why:** MCP server giving the MBPP agent a way to test its solution.

- `mcp = MCPServer("mbpp-tools")` (mcp v2 API).
- **`run_tests(code: str, test_list: list[str]) -> str`** — writes `code + tests` to a temp file, runs it in a separate Python subprocess with a **10 s timeout**, deletes the file, returns JSON `{"success": bool, "output": str}`. Empty successful output becomes `"All test assertions passed."`.
- `__main__` — `--transport stdio|streamable-http` (default `stdio`), `--port` (default `8000`).

Running in a subprocess means the real tests are isolated from the sandbox's restrictions and an infinite loop can't hang the agent.

---

### `mcp_tools_swebench.py`

**Why:** MCP server with the mandatory SWE-bench tools for exploring, editing
and testing a repository. Locates the repo via **`TESTBED_PATH`**.

Constants: `MAX_SEARCH_RESULTS = 500`, `DEFAULT_CMD_TIMEOUT = 120` s.

Helpers: `_testbed_path()` (reads env var, errors if unset), `_resolve(path)`
(relative paths are relative to the testbed), `_run(cmd, cwd, timeout)` (returns
`(exit_code, stdout, stderr)`, never raises), `_grep(pattern, file_pattern)`
(recursive regex search, output `"/abs/path.py:<line> <content>"`).

| Tool | Parameters | Returns / behaviour |
|---|---|---|
| `read_file` | `filepath`, `start_line=1`, `end_line=None` | `"N: line"` per line; range validation errors as text |
| `edit_file` | `filepath`, `old_str`, `new_str` | Replaces **exactly one** occurrence; rejects 0 or >1 matches; for `.py` files rejects edits that cause a `SyntaxError` (file left untouched) |
| `list_files` | `directory="."`, `pattern="*"` | Sorted absolute paths (`**/*.py` for recursive) |
| `search_code` | `pattern`, `file_pattern="*.py"` | grep-style regex matches (max 500) |
| `search_function_or_class_definition_in_code` | `name` | Lines matching `def name` / `class name` |
| `find_references` | `name`, `filepath=""`, `line=0` | Whole-word text matches (the last two args are accepted for interface parity but unused) |
| `run_tests` | — | Runs `EVAL_SCRIPT` env var or `$TESTBED_PATH/eval.sh` (900 s timeout); JSON `{exit_code, stdout, stderr}` |
| `get_patch` | — | `git -c core.fileMode=false diff` in the testbed (the final answer for SWE-bench) |
| `run_command` | `command`, `workdir=""` | Shell command (120 s timeout); JSON `{stdout, stderr, exit_code}` |

Also exposes an MCP **resource** `swebench://tool-docs` (`tool_docs()`, a text list
of tools) and an MCP **prompt** `solve_swebench_task(problem_statement)`.

`__main__` — same `--transport` / `--port` flags as the MBPP server.

---

### `agent_cli_common.py`

**Why:** shared code for both CLIs so they stay thin.

- `_load_dotenv_if_present()` — runs at import; loads `./.env` via python-dotenv if it exists (warns if dotenv is missing).
- `add_common_args(parser)` — adds `--task-file`, `--output`, `--model-name`, `--provider-url`, `--api-key-env` (repeatable), `--max-iterations`, `--sandbox-config`.
- `load_task(path, model_cls)` — JSON → pydantic task model.
- `build_provider(model_name, provider_url, api_key_envs)` — reads each named env var (default `LLM_API_KEY`), warns about missing ones, errors if **none** are set, and returns a single-provider `ProviderRouter` (multiple keys → rotation).
- `write_solution(output_path, solution)` — writes pretty JSON, creating parent dirs.
- `make_error_solution(task_id, benchmark, error)` — a valid failing `SolutionOutput` so a crash still produces `solution.json`.
- `print_summary(solution)` — one-line status + token/time stats to stderr.

---

### `agent_mbpp.py`

**Why:** the MBPP command-line agent.

- Defaults (from the subject, Section 5.1.1): `DEFAULT_MAX_ITERATIONS=10`, `DEFAULT_MAX_INPUT_TOKENS=6000`, `DEFAULT_MAX_OUTPUT_TOKENS=1500`, `DEFAULT_MAX_OUTPUT_TOKENS_PER_STEP=512`, `DEFAULT_TIMEOUT_SECONDS=120`.
- `MCP_SERVER_SCRIPT` — path to `mcp_tools_mbpp.py` next to this file.
- `MBPPTaskInput` — `task_id: int`, `task_definition: str`, `function_definition: str`, `test_imports: list[str] = []`, `test_list: list[str] = []`.
- `MBPP_EXAMPLE` — a short example (write code → `run_tests` → `final_answer(code)`), kept small because the system prompt is resent every step and counts against the 6000-token budget.
- `build_task_prompt(task)` — task description, signature, test imports, the exact tests, and instructions to verify with `run_tests` then `final_answer(function_code)`.
- `main()` — parse args → load task → build `Sandbox` → launch `mcp_tools_mbpp.py` over stdio with the current Python → attach tools → build provider → run `AgentLoop` → write solution. Any exception becomes an error solution; the MCP connector is always closed.

---

### `agent_swebench.py`

**Why:** the SWE-bench command-line agent. Uses "approach (b)": the agent and
sandbox run on the **host** (so LLM calls work normally), and the MCP server runs
**inside the task's Docker container** via `docker exec`.

- Defaults (Section 5.1.2): `DEFAULT_MAX_ITERATIONS=30`, `DEFAULT_MAX_INPUT_TOKENS=300_000`, `DEFAULT_MAX_OUTPUT_TOKENS=10_000`, `DEFAULT_TIMEOUT_SECONDS=900`.
- `CONTAINER_MCP_PYTHON="/opt/miniconda3/bin/python"` — interpreter used inside SWE-bench images (base Miniconda, so the repo's own test env is untouched).
- `CONTAINER_TESTBED_PATH="/testbed"`.
- `SWEBenchTaskInput` — `instance_id`, `problem_statement`, `docker_image`, `eval_script`, `hints_text=""`, `repo=""`.
- `build_task_prompt(task)` — repo name, issue text, optional hints, and the recommended workflow ending in `final_answer(get_patch())`.

#### `DockerTestbed(docker_image, eval_script, instance_id)`

- `container_name` — `agent-smith-<instance_id>-<random8>`.
- `start()` — `docker pull` → `docker run -d --rm ... tail -f /dev/null` (kept alive; network left on so the agent may `pip install` helpers) → registers a **SIGTERM handler** for cleanup → copies `mcp_tools_swebench.py` to `/` → `pip install mcp>=2.2.0` into the container's Miniconda Python → copies `eval_script` to `/testbed/eval.sh` and `chmod +x`.
- `mcp_stdio_command()` — `docker exec -i -e TESTBED_PATH=/testbed <container> /opt/miniconda3/bin/python /mcp_tools_swebench.py`.
- `_handle_sigterm()` / `stop()` — `docker rm -f` the container and restore the previous handler; guarantees cleanup even when the grader kills the process.

#### `main()`

Like MBPP, plus: with `--testbed-path` it skips Docker and launches the MCP server
on the host with `TESTBED_PATH` set; otherwise it checks `docker` is on PATH and
uses `DockerTestbed`. The container and connector are always torn down in `finally`.

---

### `test_mbpp.py`

Direct checks of `mcp_tools_mbpp.run_tests`: a correct solution (passes), a wrong
one (fails), and an infinite loop (fails via the 10 s timeout). Run with
`uv run python test_mbpp.py`.

### `test_swebench.py`

Builds a disposable git repo with a bug (`add` returns `a - b`) and an `eval.sh`, then:

1. **Direct tests (default)** — imports the tool functions and checks `read_file`, `list_files`, `search_code`, definition/reference search, `run_command`, `get_patch` (empty before, correct diff after), `run_tests` (fails before fix, passes after), `edit_file` success / not-found / ambiguous / syntax-error rejection, and the `tool_docs` resource.
2. **MCP client tests (`--mcp`)** — launches the server over stdio, checks all 9 mandatory tools are advertised, and calls `read_file` and `get_patch` through the protocol.

Prints `N passed, M failed`; exit code `1` if any failed.

### `tools/mock_moulinette.py`

**Why:** a local stand-in for 42's grading tool so you can dry-run
`dump → run agent → validate` end to end. Passing it is **not** a guarantee of
passing the real grader.

- **Fixtures:** `MBPP_FIXTURES` (3 tasks: `add`, `is_prime`, `reverse_string`) and `SWEBENCH_FIXTURE` (`fixture__mathutils-1`, image `python:3.10-slim`).
- **`dump mbpp --output PATH [--task-id N]`** — writes one fixture (random if no id).
- **`dump swebench --output PATH`** — writes the task and builds a buggy git repo at `<output dir>/fixture_repo` (use it with `--testbed-path`).
- **`validate mbpp TASK SOLUTION`** — checks benchmark/task_id, limits (`MBPP_LIMITS`: 10 iterations, 6000 in, 1500 out, 120 s), required step fields and `system_prompt`, then **runs the submitted code against the real tests**.
- **`validate swebench TASK SOLUTION`** — same checks with `SWEBENCH_LIMITS` (30, 300000, 10000, 900 s), then applies the patch to a **fresh copy** of `fixture_repo` (`git apply --check` first) and runs `eval.sh` in Docker if available (`_run_eval_in_docker`), otherwise on the host.
- `_report()` prints `PASS`/`FAIL` with each problem; exit code `0`/`1`.

### `cache/` and `graphify-out/`

Both git-ignored.
- `cache/` — task dumps and solutions produced by runs (e.g. `mbpp_task.json`, `mbpp_solution.json`, `fixture_repo/`).
- `graphify-out/` — output of the local `graphify` knowledge-graph tool; not used by the program.

---

## 6. Data formats

### MBPP task (`--task-file`)

```json
{
  "task_id": 129,
  "task_definition": "Write a function to calculate magic square.",
  "function_definition": "def magic_square_test(my_matrix):",
  "test_imports": [],
  "test_list": ["assert magic_square_test([[2,7,6],[9,5,1],[4,3,8]])==True"]
}
```

### SWE-bench task (`--task-file`)

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

### Solution (`--output`)

```json
{
  "task_id": "129",
  "benchmark": "mbpp",
  "success": true,
  "solution": "def magic_square_test(my_matrix): ...",
  "iterations": 2,
  "total_requests": 2,
  "total_input_tokens": 2100,
  "total_output_tokens": 340,
  "total_time_seconds": 5.2,
  "steps": [ { "step": 1, "input_tokens": 930, "output_tokens": 200, "request_time_ms": 1800.0,
               "timestamp": "...", "api_url": "...", "model_name": "...", "llm_output": "...",
               "sandbox_input": "...", "sandbox_output": "...", "retries": 0 } ],
  "system_prompt": "You are an autonomous coding agent...",
  "error": null,
  "timestamp": "..."
}
```

---

## 7. Limits and defaults cheat sheet

| Setting | MBPP | SWE-bench | Overridable? |
|---|---|---|---|
| Max iterations | 10 | 30 | `--max-iterations` (both) |
| Cumulative input tokens | 6 000 | 300 000 | MBPP only (`--max-input-tokens`) |
| Cumulative output tokens | 1 500 | 10 000 | MBPP only (`--max-output-tokens`) |
| Output tokens per step | 512 | 512 | MBPP only (`--max-output-tokens-per-step`) |
| Wall-clock timeout | 120 s | 900 s | MBPP only (`--timeout-seconds`) |
| Sandbox step timeout | 30 s | 30 s | `--sandbox-config` |
| Sandbox memory | 512 MB | 512 MB | `--sandbox-config` |
| Sandbox output cap | 8 000 chars | 8 000 chars | no (constant) |
| MBPP `run_tests` timeout | 10 s | — | no |
| SWE-bench `run_tests` timeout | — | 900 s | no |
| SWE-bench `run_command` timeout | — | 120 s | no |
| MCP tool-call timeout (client side) | 120 s | 120 s | no |
| LLM HTTP timeout | 60 s | 60 s | provider config only |
| LLM temperature | 0.0 | 0.0 | no |
| Stop sequence | `<end_code>` | `<end_code>` | no (CLI) |

---

## 8. Known limitations

(Also documented at the bottom of `sandbox.py`.)

1. The sandbox is an exclusion list over real Python builtins, not a true capability sandbox — attribute-chasing tricks (e.g. via `__subclasses__()`) could still escape it.
2. Timeout and memory limits are POSIX-only (`SIGALRM`, `RLIMIT_AS`), and they apply to the whole agent process while a step runs.
3. Network access is blocked only indirectly: `socket` and friends are not in the import allowlist.
4. MCP tool calls run **outside** the sandbox's restrictions (they are a separate security domain); they are bounded only by the client's 120 s timeout and the server's own timeouts.
5. Client and servers must both use `mcp>=2`; mixing v1 and v2 fails at the protocol level.
6. `find_references` is a plain text search, not semantic analysis.
7. Token counting for the budget check is an estimate (~3 chars/token); the real numbers come from the provider's `usage` field.
