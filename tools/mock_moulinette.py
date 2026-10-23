"""
mock_moulinette.py — a LOCAL STAND-IN for the real `moulinette_eval` tool.

The actual moulinette is 42's own grading harness -- this is not it, and
passing here is not a guarantee of passing the real thing. What this DOES
give you: a way to dry-run your full `dump -> run agent -> validate` cycle
end to end, against the exact limits and pass criteria the subject
documents (Sections 5.1-5.3), using small built-in fixture tasks instead of
the real MBPP/SWE-bench datasets.

Usage (mirrors the real CLI shape from the subject):

    python mock_moulinette.py dump mbpp --output cache/mbpp_task.json [--task-id 1]
    python mock_moulinette.py validate mbpp cache/mbpp_task.json cache/mbpp_solution.json

    python mock_moulinette.py dump swebench --output cache/swebench_task.json
    python mock_moulinette.py validate swebench cache/swebench_task.json cache/swebench_solution.json

SWE-bench validate note: without Docker, this can only check structural
correctness (does the patch apply cleanly, are the limits respected) -- it
cannot run the real eval_script's hidden tests in isolation. If Docker is
available on PATH, it additionally builds/reuses the task's fixture repo,
applies the patch for real, and runs eval.sh inside a throwaway container.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixture task bank (stand-ins for the real datasets, which this tool has no
# access to -- swap these for real moulinette-provided tasks once you have it)
# ---------------------------------------------------------------------------

MBPP_FIXTURES = [
    {
        "task_id": 1,
        "task_definition": "Write a function to add two numbers.",
        "function_definition": "def add(a, b):\n    pass",
        "test_imports": [],
        "test_list": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"],
    },
    {
        "task_id": 2,
        "task_definition": "Write a function to check if a number is prime.",
        "function_definition": "def is_prime(n):\n    pass",
        "test_imports": [],
        "test_list": [
            "assert is_prime(2) == True",
            "assert is_prime(4) == False",
            "assert is_prime(17) == True",
            "assert is_prime(1) == False",
        ],
    },
    {
        "task_id": 3,
        "task_definition": "Write a function to reverse a string.",
        "function_definition": "def reverse_string(s):\n    pass",
        "test_imports": [],
        "test_list": [
            "assert reverse_string('hello') == 'olleh'",
            "assert reverse_string('') == ''",
        ],
    },
]

# A minimal SWE-bench-style fixture: a tiny git repo built on the fly, with
# a real bug and a real (fast) eval script. docker_image points at a
# lightweight public Python image so `docker pull` succeeds if you have
# Docker -- there's no real SWE-bench Docker image here since this is a
# fixture, not a real dataset entry.
SWEBENCH_FIXTURE = {
    "instance_id": "fixture__mathutils-1",
    "problem_statement": "The add() function in mathutils.py returns the wrong result -- it subtracts instead of adding.",
    "docker_image": "python:3.10-slim",
    "hints_text": "",
    "repo": "fixture/mathutils",
}


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


def cmd_dump_mbpp(args) -> int:
    task_id = args.task_id
    if task_id is not None:
        matches = [t for t in MBPP_FIXTURES if t["task_id"] == task_id]
        if not matches:
            print(f"No fixture MBPP task with task_id={task_id}.", file=sys.stderr)
            return 1
        task = matches[0]
    else:
        import random

        task = random.choice(MBPP_FIXTURES)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(task, indent=2))
    print(f"Dumped MBPP fixture task_id={task['task_id']} to {out}")
    return 0


def _build_swebench_fixture_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "mathutils.py").write_text(
        "def add(a, b):\n"
        "    return a - b  # BUG: should add, not subtract\n"
    )
    (repo_dir / "eval.sh").write_text(
        "#!/bin/bash\n"
        "python3 -c \"from mathutils import add; assert add(2, 3) == 5; print('PASS')\"\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@test.local"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial (buggy)"], cwd=repo_dir, check=True)


def cmd_dump_swebench(args) -> int:
    task = dict(SWEBENCH_FIXTURE)
    task["eval_script"] = (
        "#!/bin/bash\n"
        "python3 -c \"from mathutils import add; assert add(2, 3) == 5; print('PASS')\"\n"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(task, indent=2))
    print(f"Dumped SWE-bench fixture task_id={task['instance_id']} to {out}")

    # Also materialize the fixture repo next to the task file, since your
    # agent (in --testbed-path mode, or a real Docker image you build
    # yourself from it) needs an actual repo to point at.
    fixture_repo = out.parent / "fixture_repo"
    if fixture_repo.exists():
        shutil.rmtree(fixture_repo)
    _build_swebench_fixture_repo(fixture_repo)
    print(f"Materialized the fixture repo at {fixture_repo} (use --testbed-path for a Docker-free agent run).")
    return 0


# ---------------------------------------------------------------------------
# validate: shared limit checks (Section 5.1)
# ---------------------------------------------------------------------------

MBPP_LIMITS = dict(max_iterations=10, max_input_tokens=6000, max_output_tokens=1500, timeout_seconds=120)
SWEBENCH_LIMITS = dict(max_iterations=30, max_input_tokens=300_000, max_output_tokens=10_000, timeout_seconds=900)


def _check_limits(solution: dict, limits: dict) -> list[str]:
    problems = []
    if solution["iterations"] > limits["max_iterations"]:
        problems.append(f"iterations {solution['iterations']} exceeds max_iterations {limits['max_iterations']}")
    if solution["total_input_tokens"] > limits["max_input_tokens"]:
        problems.append(
            f"total_input_tokens {solution['total_input_tokens']} exceeds max_input_tokens {limits['max_input_tokens']}"
        )
    if solution["total_output_tokens"] > limits["max_output_tokens"]:
        problems.append(
            f"total_output_tokens {solution['total_output_tokens']} exceeds max_output_tokens {limits['max_output_tokens']}"
        )
    if solution["total_time_seconds"] > limits["timeout_seconds"]:
        problems.append(
            f"total_time_seconds {solution['total_time_seconds']:.1f} exceeds timeout {limits['timeout_seconds']}s"
        )
    return problems


def _check_required_fields(solution: dict) -> list[str]:
    """Section 4.1.1 / 4.4: every step must carry api_url, model_name,
    llm_output, sandbox_input, sandbox_output, retries; SolutionOutput
    must carry system_prompt."""
    problems = []
    if not solution.get("system_prompt", "").strip():
        problems.append("system_prompt is empty -- Section 4.4 requires the full prompt for provenance checking")
    for i, step in enumerate(solution.get("steps", []), start=1):
        for field in ("api_url", "model_name", "llm_output", "sandbox_input", "sandbox_output"):
            if field not in step:
                problems.append(f"step {i}: missing required field '{field}'")
        if "retries" not in step:
            problems.append(f"step {i}: missing required field 'retries'")
    return problems


# ---------------------------------------------------------------------------
# validate mbpp
# ---------------------------------------------------------------------------


def cmd_validate_mbpp(args) -> int:
    task = json.loads(Path(args.task_file).read_text())
    solution = json.loads(Path(args.solution_file).read_text())

    print(f"Validating MBPP task_id={task['task_id']} ...")
    problems = []

    if solution.get("benchmark") != "mbpp":
        problems.append(f"benchmark field is {solution.get('benchmark')!r}, expected 'mbpp'")
    if str(solution.get("task_id")) != str(task["task_id"]):
        problems.append(f"task_id mismatch: solution says {solution.get('task_id')!r}, task says {task['task_id']!r}")

    problems += _check_limits(solution, MBPP_LIMITS)
    problems += _check_required_fields(solution)

    if not solution.get("success"):
        problems.append(f"agent reported success=False (error: {solution.get('error')})")
    else:
        # Actually run the submitted code against the task's real test_list.
        code = solution.get("solution", "")
        script = code + "\n\n" + "\n".join(task["test_list"]) + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            path = f.name
        try:
            result = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                problems.append(f"submitted solution FAILS the real test_list:\n{result.stdout}{result.stderr}")
        finally:
            Path(path).unlink(missing_ok=True)

    return _report(problems, task_label=f"MBPP task {task['task_id']}")


# ---------------------------------------------------------------------------
# validate swebench
# ---------------------------------------------------------------------------


def cmd_validate_swebench(args) -> int:
    task = json.loads(Path(args.task_file).read_text())
    solution = json.loads(Path(args.solution_file).read_text())

    print(f"Validating SWE-bench task instance_id={task['instance_id']} ...")
    problems = []

    if solution.get("benchmark") != "swebench":
        problems.append(f"benchmark field is {solution.get('benchmark')!r}, expected 'swebench'")
    if solution.get("task_id") != task["instance_id"]:
        problems.append(f"task_id mismatch: solution says {solution.get('task_id')!r}, task says {task['instance_id']!r}")

    problems += _check_limits(solution, SWEBENCH_LIMITS)
    problems += _check_required_fields(solution)

    if not solution.get("success"):
        problems.append(f"agent reported success=False (error: {solution.get('error')})")
    else:
        patch = solution.get("solution", "")
        if not patch.strip():
            problems.append("solution field (patch) is empty despite success=True")
        else:
            problems += _apply_and_run_patch(args, task, patch)

    return _report(problems, task_label=f"SWE-bench {task['instance_id']}")


def _apply_and_run_patch(args, task: dict, patch: str) -> list[str]:
    """Applies the patch to a FRESH copy of the fixture repo (never the one
    your agent already touched) and, if Docker is available, actually runs
    eval.sh inside a container. Falls back to a bare `git apply --check`
    when Docker isn't available."""
    problems: list[str] = []
    fixture_repo = Path(args.task_file).parent / "fixture_repo"
    if not fixture_repo.is_dir():
        problems.append(
            "no fixture_repo found next to the task file -- was this task.json produced by "
            "`dump swebench`? Cannot verify the patch without it."
        )
        return problems

    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "repo"
        shutil.copytree(fixture_repo, fresh)

        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
            f.write(patch)
            patch_path = f.name

        try:
            check = subprocess.run(
                ["git", "apply", "--check", patch_path], cwd=fresh, capture_output=True, text=True
            )
            if check.returncode != 0:
                problems.append(f"patch does not apply cleanly:\n{check.stderr}")
                return problems

            subprocess.run(["git", "apply", patch_path], cwd=fresh, check=True)

            if shutil.which("docker"):
                problems += _run_eval_in_docker(fresh, task)
            else:
                # No Docker: run the fixture's eval.sh directly on the host as a
                # best-effort check. This is NOT how real grading works (real
                # tasks need their actual Docker image), it's just better than
                # nothing for this fixture.
                eval_script = fresh / "eval.sh"
                if eval_script.is_file():
                    result = subprocess.run(
                        ["bash", str(eval_script)], cwd=fresh, capture_output=True, text=True, timeout=60
                    )
                    if result.returncode != 0:
                        problems.append(
                            f"eval.sh failed after applying the patch (ran on host, no Docker available):\n"
                            f"{result.stdout}{result.stderr}"
                        )
                    print("  (ran eval.sh on the host -- no Docker available; real grading uses the actual image)")
        finally:
            Path(patch_path).unlink(missing_ok=True)

    return problems


def _run_eval_in_docker(repo_dir: Path, task: dict) -> list[str]:
    problems = []
    container = "mock-moulinette-verify"
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    try:
        pull = subprocess.run(["docker", "pull", task["docker_image"]], capture_output=True, text=True)
        if pull.returncode != 0:
            problems.append(f"docker pull failed: {pull.stderr}")
            return problems

        run = subprocess.run(
            ["docker", "run", "-d", "--name", container, "-v", f"{repo_dir}:/testbed",
             "-w", "/testbed", task["docker_image"], "tail", "-f", "/dev/null"],
            capture_output=True, text=True,
        )
        if run.returncode != 0:
            problems.append(f"docker run failed: {run.stderr}")
            return problems

        result = subprocess.run(
            ["docker", "exec", container, "bash", "/testbed/eval.sh"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            problems.append(f"eval.sh failed inside Docker:\n{result.stdout}{result.stderr}")
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)

    return problems


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _report(problems: list[str], task_label: str) -> int:
    print()
    if not problems:
        print(f"PASS -- {task_label}")
        return 0
    print(f"FAIL -- {task_label}")
    for p in problems:
        print(f"  - {p}")
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog="mock_moulinette")
    sub = parser.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump")
    dump_sub = dump.add_subparsers(dest="benchmark", required=True)
    dump_mbpp = dump_sub.add_parser("mbpp")
    dump_mbpp.add_argument("--output", required=True)
    dump_mbpp.add_argument("--task-id", type=int, default=None)
    dump_mbpp.set_defaults(func=cmd_dump_mbpp)
    dump_swe = dump_sub.add_parser("swebench")
    dump_swe.add_argument("--output", required=True)
    dump_swe.set_defaults(func=cmd_dump_swebench)

    validate = sub.add_parser("validate")
    validate_sub = validate.add_subparsers(dest="benchmark", required=True)
    val_mbpp = validate_sub.add_parser("mbpp")
    val_mbpp.add_argument("task_file")
    val_mbpp.add_argument("solution_file")
    val_mbpp.set_defaults(func=cmd_validate_mbpp)
    val_swe = validate_sub.add_parser("swebench")
    val_swe.add_argument("task_file")
    val_swe.add_argument("solution_file")
    val_swe.set_defaults(func=cmd_validate_swebench)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
