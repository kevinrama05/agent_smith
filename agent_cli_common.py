"""
agent_cli_common.py — shared plumbing for agent_mbpp.py and agent_swebench.py.

Both CLIs are thin: parse args -> load task -> build a Sandbox+MCP connection
-> build a provider -> run AgentLoop -> write solution.json. This module
holds the parts that are identical between the two.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Type, TypeVar

from pydantic import BaseModel

from llm_provider import OpenAICompatibleProvider, ProviderRouter, RoutedProvider
from agent_loop import SolutionOutput


def _load_dotenv_if_present() -> None:
    """Best-effort: if a .env file exists in the current directory and
    python-dotenv is installed, load it into os.environ. Never required --
    exam invocations pass --env-file to the harness, which sets real
    environment variables directly; this is purely a local-dev convenience
    so you don't have to `export` keys by hand every session."""
    env_path = Path(".env")
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        print(
            "[agent] Found a .env file but python-dotenv is not installed "
            "(`uv add python-dotenv`) -- not loading it automatically. "
            "Export the variables manually instead.",
            file=sys.stderr,
        )


_load_dotenv_if_present()

T = TypeVar("T", bound=BaseModel)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by both agent_mbpp and agent_swebench, matching the
    CLI usage examples in Sections 4.3 and 4.4."""
    parser.add_argument("--task-file", required=True, help="Path to the task JSON dumped by moulinette_eval.")
    parser.add_argument("--output", required=True, help="Path to write solution.json to.")
    parser.add_argument("--model-name", required=True, help="Model identifier, e.g. 'qwen/qwen3-235b-a22b-2507'.")
    parser.add_argument("--provider-url", required=True, help="Base URL of the LLM API endpoint.")
    parser.add_argument(
        "--api-key-env",
        action="append",
        default=None,
        help=(
            "Name of an environment variable holding an API key for this provider. "
            "Repeat for multiple keys (enables rotation). Defaults to LLM_API_KEY "
            "if not given -- students should point this at their provider's "
            "standard variable, e.g. --api-key-env OPENROUTER_API_KEY."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="Override the default iteration limit.")
    parser.add_argument("--sandbox-config", default=None, help="Path to a SandboxConfig JSON file (optional).")


def load_task(path: str, model_cls: Type[T]) -> T:
    with open(path) as f:
        data = json.load(f)
    return model_cls(**data)


def build_provider(model_name: str, provider_url: str, api_key_envs: List[str] | None) -> ProviderRouter:
    """Builds a single-provider ProviderRouter from CLI args, per the exam's
    `--model-name`/`--provider-url` invocation shape (Sections 4.3/4.4).
    Returning a ProviderRouter (rather than the bare OpenAICompatibleProvider)
    keeps AgentLoop's call signature identical regardless of whether it's
    driven by this CLI or by a full multi-provider benchmarking config --
    AgentLoop never needs to know or pass a `model` argument itself.
    Still supports multi-key rotation via repeated --api-key-env flags,
    satisfying Section 4.6.1's multi-token requirement even for a single
    provider/model pair.

    For your own multi-provider benchmarking runs (Section 4.7), use
    llm_provider.load_provider_router() directly instead of this function --
    it's built for the graded CLI's single-provider invocation shape.
    """
    env_names = api_key_envs or ["LLM_API_KEY"]
    keys = []
    missing = []
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            keys.append(value)
        else:
            missing.append(env_name)

    if not keys:
        raise ValueError(
            f"No API key found. Checked environment variable(s): {', '.join(env_names)}. "
            f"Set one of these, or pass --api-key-env to point at the right variable."
        )
    if missing:
        print(
            f"[agent] Warning: env var(s) not set, skipping: {', '.join(missing)}. "
            f"Continuing with {len(keys)} key(s).",
            file=sys.stderr,
        )

    provider = OpenAICompatibleProvider(name=model_name, base_url=provider_url, api_keys=keys)
    return ProviderRouter([RoutedProvider(provider=provider, model=model_name)])


def write_solution(output_path: str, solution: SolutionOutput) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(solution.model_dump_json(indent=2))


def make_error_solution(task_id: str, benchmark: str, error: str) -> SolutionOutput:
    """A valid, well-formed SolutionOutput for the 'something failed before
    we could even run the loop' case (bad task file, missing API key,
    Docker unavailable, etc.). Per Section 3.1 ('crashes during evaluation
    will result in failure'), the CLI must always produce a solution.json --
    even a failing one -- rather than exiting via an uncaught traceback."""
    return SolutionOutput(
        task_id=task_id,
        benchmark=benchmark,
        success=False,
        solution="",
        iterations=0,
        total_requests=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_time_seconds=0.0,
        steps=[],
        system_prompt="",
        error=error,
    )


def print_summary(solution: SolutionOutput) -> None:
    status = "SUCCESS" if solution.success else "FAILED"
    print(f"[agent] {status} -- task {solution.task_id} ({solution.benchmark})", file=sys.stderr)
    print(
        f"[agent] iterations={solution.iterations} "
        f"input_tokens={solution.total_input_tokens} "
        f"output_tokens={solution.total_output_tokens} "
        f"time={solution.total_time_seconds:.1f}s",
        file=sys.stderr,
    )
    if solution.error:
        print(f"[agent] error: {solution.error}", file=sys.stderr)