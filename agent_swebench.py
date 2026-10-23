"""
agent_swebench.py — SWE-bench agent CLI (Section 4.4).

    uv run python -m agent_swebench --task-file ../cache/swebench_task.json \\
      --output ../cache/swebench_solution.json \\
      --model-name "model/name" --provider-url "https://provider.api/v1"

Uses approach (b) from Section 4.4: the sandbox and orchestrator run on the
host (so the LLM provider calls just work), and MCP tools bridge into the
task's Docker container via `docker exec`. The container is started fresh
for this task and torn down afterward, including on a forced kill
(SIGTERM), per Section 5.1's cleanup requirement.

For local development without Docker, pass --testbed-path pointing at any
git repo on disk -- this skips all container orchestration and runs
mcp_tools_swebench.py directly against that path instead. Real evaluation
always goes through the Docker path.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from agent_cli_common import add_common_args, build_provider, load_task, make_error_solution, print_summary, write_solution
from agent_loop import AgentLoop
from sandbox import MCPToolConnector, Sandbox, SandboxConfig

# Section 5.1.2
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_MAX_INPUT_TOKENS = 300_000
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
DEFAULT_TIMEOUT_SECONDS = 900

MCP_SERVER_SCRIPT = Path(__file__).resolve().parent / "mcp_tools_swebench.py"
CONTAINER_MCP_PYTHON = "/opt/miniconda3/bin/python"
CONTAINER_TESTBED_PATH = "/testbed"


class SWEBenchTaskInput(BaseModel):
    """Input for a SWE-bench task, provided by the moulinette."""

    instance_id: str
    problem_statement: str
    docker_image: str
    eval_script: str
    hints_text: str = Field(default="")
    repo: str = Field(default="")


def build_task_prompt(task: SWEBenchTaskInput) -> str:
    hints_block = f"\n## Hints\n\n{task.hints_text}\n" if task.hints_text else ""
    return f"""## Repository

{task.repo or "(unspecified)"}

## Issue to fix

{task.problem_statement}
{hints_block}
Explore the repository with search_code / search_function_or_class_definition_in_code,
read relevant files with read_file, make changes with edit_file, and verify with
run_tests() as you go. Once run_tests() shows the fix works, call
final_answer(get_patch()) with the resulting patch.
"""


# ---------------------------------------------------------------------------
# Docker orchestration (approach b: sandbox on host, MCP bridges via `docker exec`)
# ---------------------------------------------------------------------------


class DockerTestbed:
    """Pulls the task's image, runs it, copies the MCP tool file and eval
    script in, and guarantees cleanup -- including on SIGTERM, per
    Section 5.1's "make sure your Docker container cleanup can still run"
    requirement."""

    def __init__(self, docker_image: str, eval_script: str, instance_id: str):
        self.docker_image = docker_image
        self.eval_script = eval_script
        self.container_name = f"agent-smith-{instance_id}-{uuid.uuid4().hex[:8]}"
        self._started = False
        self._old_sigterm_handler = None

    def _run(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, **kwargs)

    def start(self) -> None:
        pull = self._run(["docker", "pull", self.docker_image])
        if pull.returncode != 0:
            raise RuntimeError(f"docker pull failed for {self.docker_image}:\n{pull.stderr}")

        # Note: NOT using --network none here, even though nothing in this
        # container needs outbound network for the agent's own operation
        # (LLM calls happen on the host). Section 4.4 explicitly allows
        # installing extra dependencies inside the container (ruff, jedi,
        # tree, etc.) via run_command, which needs network access to do so.
        run = self._run([
            "docker", "run", "-d", "--rm", "--name", self.container_name,
            self.docker_image, "tail", "-f", "/dev/null",
        ])
        if run.returncode != 0:
            raise RuntimeError(f"docker run failed:\n{run.stderr}")
        self._started = True

        # Register cleanup on a forced kill (SIGTERM from the moulinette's
        # timeout enforcement), not just normal exit.
        self._old_sigterm_handler = signal.signal(signal.SIGTERM, self._handle_sigterm)

        cp_tools = self._run(["docker", "cp", str(MCP_SERVER_SCRIPT), f"{self.container_name}:/mcp_tools_swebench.py"])
        if cp_tools.returncode != 0:
            raise RuntimeError(f"docker cp of MCP tools failed:\n{cp_tools.stderr}")

        # The MCP server runs inside the container under the Miniconda base
        # interpreter (Python 3.11), which doesn't ship with `mcp`. The testbed
        # env is left untouched so the repo's own tests aren't affected.
        pip = self._run([
            "docker", "exec", self.container_name,
            CONTAINER_MCP_PYTHON, "-m", "pip", "install", "-q", "mcp>=2.2.0",
        ])
        if pip.returncode != 0:
            raise RuntimeError(f"pip install of mcp in container failed:\n{pip.stderr}")

        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write(self.eval_script)
            eval_local_path = f.name
        try:
            cp_eval = self._run([
                "docker", "cp", eval_local_path, f"{self.container_name}:{CONTAINER_TESTBED_PATH}/eval.sh"
            ])
            if cp_eval.returncode != 0:
                raise RuntimeError(f"docker cp of eval script failed:\n{cp_eval.stderr}")
        finally:
            Path(eval_local_path).unlink(missing_ok=True)

        self._run(["docker", "exec", self.container_name, "chmod", "+x", f"{CONTAINER_TESTBED_PATH}/eval.sh"])

    def _handle_sigterm(self, signum, frame) -> None:
        self.stop()
        if callable(self._old_sigterm_handler):
            self._old_sigterm_handler(signum, frame)
        else:
            sys.exit(1)

    def mcp_stdio_command(self) -> str:
        # docker exec doesn't inherit the host env, so TESTBED_PATH must be
        # set explicitly for the MCP server inside the container.
        return (
            f"docker exec -i -e TESTBED_PATH={CONTAINER_TESTBED_PATH} "
            f"{self.container_name} {CONTAINER_MCP_PYTHON} /mcp_tools_swebench.py"
        )

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True, text=True)
        if self._old_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._old_sigterm_handler)


def salvage_patch(sandbox: Sandbox, solution):
    """The loop stopped without final_answer (token or iteration budget), but
    the agent's edits are still in the container. Submit them instead of an
    empty patch: the moulinette validates the patch, not the success flag."""
    try:
        result = sandbox.execute("final_answer(get_patch())")
    except Exception as e:
        print(f"[agent] could not salvage patch: {e}", file=sys.stderr)
        return solution
    patch = result.final_answer_value if result.is_final_answer else None
    if not isinstance(patch, str) or not patch.strip() or patch.startswith("Error"):
        return solution
    print("[agent] no final_answer: submitting the current git diff as the patch.", file=sys.stderr)
    return solution.model_copy(update={"solution": patch})


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent_swebench")
    add_common_args(parser)
    parser.add_argument(
        "--testbed-path",
        default=None,
        help="LOCAL DEV ONLY: skip Docker and run mcp_tools_swebench.py directly against this path.",
    )
    args = parser.parse_args()

    task_id_for_errors = args.task_file
    testbed: Optional[DockerTestbed] = None
    connector: Optional[MCPToolConnector] = None

    try:
        task = load_task(args.task_file, SWEBenchTaskInput)
        task_id_for_errors = task.instance_id
        task_prompt = build_task_prompt(task)

        sandbox_config = SandboxConfig() if not args.sandbox_config else SandboxConfig.model_validate_json(
            Path(args.sandbox_config).read_text()
        )
        sandbox = Sandbox(sandbox_config)

        if not MCP_SERVER_SCRIPT.is_file():
            raise RuntimeError(f"expected MCP server at {MCP_SERVER_SCRIPT}, not found.")

        if args.testbed_path:
            print(f"[agent] --testbed-path given: skipping Docker, using {args.testbed_path} directly.", file=sys.stderr)
            connector = MCPToolConnector(
                "stdio",
                f"{sys.executable} {MCP_SERVER_SCRIPT}",
                extra_env={"TESTBED_PATH": str(Path(args.testbed_path).resolve())},
            )
        else:
            if shutil.which("docker") is None:
                raise RuntimeError("docker is not available on PATH.")
            testbed = DockerTestbed(task.docker_image, task.eval_script, task.instance_id)
            testbed.start()
            connector = MCPToolConnector("stdio", testbed.mcp_stdio_command())

        connector.connect()
        sandbox.attach_mcp(connector)

        provider = build_provider(args.model_name, args.provider_url, args.api_key_env)

        loop = AgentLoop(
            provider=provider,
            sandbox=sandbox,
            max_iterations=args.max_iterations or DEFAULT_MAX_ITERATIONS,
            max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        )
        solution = loop.run(task_prompt=task_prompt, task_id=task_id_for_errors, benchmark="swebench")
        if not solution.success:
            solution = salvage_patch(sandbox, solution)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        # Section 3.1: "crashes during evaluation will result in failure" --
        # always produce a valid solution.json, never an uncaught traceback.
        print(f"[agent] ERROR: {e}", file=sys.stderr)
        solution = make_error_solution(task_id_for_errors, "swebench", str(e))
    finally:
        if connector is not None:
            connector.close()
        if testbed is not None:
            testbed.stop()

    write_solution(args.output, solution)
    print_summary(solution)
    # A written solution.json is a normal exit, even for a failed task: a
    # non-zero code makes the moulinette's run-agent abort before validation.
    return 0


if __name__ == "__main__":
    sys.exit(main())