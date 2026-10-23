"""
agent_mbpp.py — MBPP agent CLI (Section 4.3).

    uv run python -m agent_mbpp --task-file ../cache/mbpp_task.json \\
      --output ../cache/mbpp_solution.json \\
      --model-name "model/name" --provider-url "https://provider.api/v1"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from agent_cli_common import add_common_args, build_provider, load_task, make_error_solution, print_summary, write_solution
from agent_loop import AgentLoop
from sandbox import MCPToolConnector, Sandbox, SandboxConfig

# Section 5.1.1
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_INPUT_TOKENS = 6000
DEFAULT_MAX_OUTPUT_TOKENS = 1500
DEFAULT_MAX_OUTPUT_TOKENS_PER_STEP = 512
DEFAULT_TIMEOUT_SECONDS = 120

MCP_SERVER_SCRIPT = Path(__file__).resolve().parent / "mcp_tools_mbpp.py"


class MBPPTaskInput(BaseModel):
    """Input for MBPP task evaluation."""

    task_id: int
    task_definition: str
    function_definition: str
    test_imports: List[str] = Field(default_factory=list)
    test_list: List[str] = Field(default_factory=list)


# Kept short and MBPP-specific: the whole system prompt is resent on every
# step and counts against the 6000 cumulative input-token budget.
MBPP_EXAMPLE = """--- Example ---

Thought: I'll implement the function and run the given tests in one step.
Code:
```python
code = \'\'\'
def add(a, b):
    return a + b
\'\'\'
print(run_tests(code=code, test_list=["assert add(2, 3) == 5"]))
```
<end_code>

(Observation: {"success": true, "output": "All test assertions passed."})

Thought: Tests pass, submitting.
Code:
```python
final_answer(code)
```
<end_code>

"""


def build_task_prompt(task: MBPPTaskInput) -> str:
    imports_block = "\n".join(task.test_imports) if task.test_imports else "(none)"
    tests_block = "\n".join(task.test_list)
    return f"""## Task

{task.task_definition}

## Function signature to implement

```python
{task.function_definition}
```

## Test imports required by the tests

{imports_block}

## Tests your solution must pass

```python
{tests_block}
```

Write a complete implementation, then call `run_tests(code=..., test_list=...)`
using the EXACT test list above to verify it. Once it passes, call
`final_answer(your_function_code)` with the final Python source of your
function (not the test code).
"""


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent_mbpp")
    add_common_args(parser)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=f"Cumulative output token budget for the whole run (default: {DEFAULT_MAX_OUTPUT_TOKENS})",
    )
    parser.add_argument(
        "--max-output-tokens-per-step",
        type=int,
        default=None,
        help=f"Per-step output token cap, clamped further by remaining budget "
             f"(default: {DEFAULT_MAX_OUTPUT_TOKENS_PER_STEP})",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help=f"Cumulative input token budget for the whole run (default: {DEFAULT_MAX_INPUT_TOKENS})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help=f"Wall-clock timeout in seconds for the whole run (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args()

    # task_id may not be known yet if the task file itself fails to load --
    # fall back to the raw path so the error solution.json is still identifiable.
    task_id_for_errors = args.task_file
    connector = None
    try:
        task = load_task(args.task_file, MBPPTaskInput)
        task_id_for_errors = str(task.task_id)
        task_prompt = build_task_prompt(task)

        sandbox_config = SandboxConfig() if not args.sandbox_config else SandboxConfig.model_validate_json(
            Path(args.sandbox_config).read_text()
        )
        sandbox = Sandbox(sandbox_config)

        if not MCP_SERVER_SCRIPT.is_file():
            raise RuntimeError(f"expected MCP server at {MCP_SERVER_SCRIPT}, not found.")

        connector = MCPToolConnector("stdio", f"{sys.executable} {MCP_SERVER_SCRIPT}")
        connector.connect()
        sandbox.attach_mcp(connector)

        provider = build_provider(args.model_name, args.provider_url, args.api_key_env)

        loop = AgentLoop(
            provider=provider,
            sandbox=sandbox,
            max_iterations=args.max_iterations or DEFAULT_MAX_ITERATIONS,
            max_input_tokens=args.max_input_tokens or DEFAULT_MAX_INPUT_TOKENS,
            max_output_tokens=args.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
            max_output_tokens_per_step=args.max_output_tokens_per_step or DEFAULT_MAX_OUTPUT_TOKENS_PER_STEP,
            timeout_seconds=args.timeout_seconds or DEFAULT_TIMEOUT_SECONDS,
            example=MBPP_EXAMPLE,
        )
        solution = loop.run(task_prompt=task_prompt, task_id=task_id_for_errors, benchmark="mbpp")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        # Section 3.1: "crashes during evaluation will result in failure" --
        # always produce a valid solution.json, never an uncaught traceback.
        print(f"[agent] ERROR: {e}", file=sys.stderr)
        solution = make_error_solution(task_id_for_errors, "mbpp", str(e))
    finally:
        if connector is not None:
            connector.close()

    write_solution(args.output, solution)
    print_summary(solution)
    return 0 if solution.success else 1


if __name__ == "__main__":
    sys.exit(main())