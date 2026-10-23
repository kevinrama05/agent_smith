"""
mcp_tools_swebench.py — MCP server exposing the mandatory SWE-bench tools
(Section 4.5). Locates the repository via the TESTBED_PATH env var.

    python mcp_tools_swebench.py [--transport stdio|streamable-http] [--port 8000]
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("swebench-tools")

MAX_SEARCH_RESULTS = 500
DEFAULT_CMD_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _testbed_path() -> Path:
    """Locate the repository root via the TESTBED_PATH env var, per Section 4.4."""
    raw = os.environ.get("TESTBED_PATH")
    if not raw:
        raise RuntimeError("TESTBED_PATH environment variable is not set")
    return Path(raw)


def _resolve(filepath: str) -> Path:
    """Resolve a path: absolute paths pass through, relative paths are
    interpreted relative to TESTBED_PATH."""
    p = Path(filepath)
    return p if p.is_absolute() else _testbed_path() / p


def _run(cmd, cwd=None, timeout: int = DEFAULT_CMD_TIMEOUT):
    """Run a command (list or shell string), returning (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return -1, stdout, f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", f"Failed to execute command: {e}"


# ---------------------------------------------------------------------------
# File System Tools (Section 4.5.1)
# ---------------------------------------------------------------------------

@mcp.tool()
def read_file(filepath: str, start_line: int = 1, end_line: int | None = None) -> str:
    """Read a file's content with line numbers, similar to `cat -n`.

    Args:
        filepath: Absolute path, or a path relative to TESTBED_PATH.
        start_line: First line to include (1-indexed). Defaults to 1.
        end_line: Last line to include (1-indexed, inclusive). Defaults to
            the end of the file.

    Returns:
        Lines formatted as "<line_number>: <line_content>", one per line.
        An "Error: ..." string is returned on failure instead of raising.
    """
    path = _resolve(filepath)
    if not path.is_file():
        return f"Error: file not found: {filepath}"

    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception as e:
        return f"Error: could not read file: {e}"

    total = len(lines)
    if total == 0:
        return "(file is empty)"
    if start_line < 1 or start_line > total:
        return f"Error: start_line {start_line} out of range (file has {total} lines)"

    end = min(end_line, total) if end_line is not None else total
    if end < start_line:
        return f"Error: end_line {end_line} is before start_line {start_line}"

    return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start_line, end + 1))


@mcp.tool()
def edit_file(filepath: str, old_str: str, new_str: str) -> str:
    """Replace an exact, unique string occurrence in a file.

    Args:
        filepath: Absolute path, or a path relative to TESTBED_PATH.
        old_str: Exact text to find and replace. Must appear exactly once
            in the file, or the edit is rejected.
        new_str: Replacement text.

    Returns:
        A success message, or an "Error: ..." message if old_str is
        missing, ambiguous (multiple matches), or the edit introduces a
        Python syntax error (checked for .py files before writing).
    """
    path = _resolve(filepath)
    if not path.is_file():
        return f"Error: file not found: {filepath}"

    try:
        content = path.read_text(errors="replace")
    except Exception as e:
        return f"Error: could not read file: {e}"

    count = content.count(old_str)
    if count == 0 and "\t" in old_str and "\t" not in content:
        # Models often write indentation as \t; the file indents with spaces.
        old_str, new_str = old_str.replace("\t", "    "), new_str.replace("\t", "    ")
        count = content.count(old_str)
    if count == 0:
        return (
            "Error: old_str not found in file. No changes made. It must match the file "
            "exactly, including indentation (spaces) -- read_file the lines first and copy them."
        )
    if count > 1:
        return (
            f"Error: old_str is not unique ({count} occurrences found). "
            "Include more surrounding context so it matches exactly one location."
        )

    new_content = content.replace(old_str, new_str, 1)

    if path.suffix == ".py":
        try:
            compile(new_content, str(path), "exec")
        except SyntaxError as e:
            return f"Error: edit introduces a syntax error, file NOT modified: {e}"

    try:
        path.write_text(new_content)
    except Exception as e:
        return f"Error: could not write file: {e}"

    return f"Successfully edited {filepath} (1 replacement)."


@mcp.tool()
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files in a directory matching a glob pattern.

    Args:
        directory: Absolute path, or a path relative to TESTBED_PATH.
        pattern: Glob pattern, e.g. "*.py" or "**/*.py" for recursive search.

    Returns:
        Newline-separated absolute file paths, sorted, or "(no files
        matched)" / an "Error: ..." message.
    """
    path = _resolve(directory)
    if not path.is_dir():
        return f"Error: directory not found: {directory}"

    try:
        matches = sorted(str(p) for p in path.glob(pattern) if p.is_file())
    except Exception as e:
        return f"Error: invalid pattern or search failed: {e}"

    return "\n".join(matches) if matches else "(no files matched)"


# ---------------------------------------------------------------------------
# Code Search Tools (Section 4.5.2)
# ---------------------------------------------------------------------------

def _grep(pattern: str, file_pattern: str) -> str:
    """Shared grep-like search used by search_code, definition search, and
    reference search. Output format matches Section 4.5.2's spec."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex: {e}"

    root = _testbed_path()
    results = []
    for filepath in sorted(root.rglob(file_pattern)):
        if not filepath.is_file():
            continue
        try:
            text = filepath.read_text(errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                results.append(f"{filepath}:{i} {line}")
                if len(results) >= MAX_SEARCH_RESULTS:
                    results.append("... (truncated, too many matches — narrow your pattern)")
                    return "\n".join(results)

    return "\n".join(results) if results else "(no matches)"


@mcp.tool()
def search_code(pattern: str, file_pattern: str = "*.py") -> str:
    """Perform a grep-like regex search across the codebase.

    Args:
        pattern: Regular expression to search for.
        file_pattern: Glob pattern restricting which files are searched
            (e.g. "*.py", "**/*.py" for recursive).

    Returns:
        Lines formatted as "/absolute/path.py:<line_number> <line_content>",
        one per match. Returns "(no matches)" if nothing is found.
    """
    return _grep(pattern, file_pattern)


@mcp.tool()
def search_function_or_class_definition_in_code(name: str) -> str:
    """Find where a function or class is defined in the codebase.

    Args:
        name: The exact function or class name to look for.

    Returns:
        Matches in the same format as search_code, restricted to lines
        that look like a `def name(...)` or `class name` declaration.
    """
    pattern = rf"^\s*(def|class)\s+{re.escape(name)}\b"
    return _grep(pattern, "*.py")


@mcp.tool()
def find_references(name: str, filepath: str = "", line: int = 0) -> str:
    """Find usages of a function or class name across the codebase.

    Args:
        name: The symbol name to search for.
        filepath: Optional context — file where the symbol is defined.
            Not used by this naive text-based implementation, but kept for
            interface parity with IDE-style "find references" tools.
        line: Optional context — line number where the symbol is defined.
            Not used by this naive implementation.

    Returns:
        Matches in the same format as search_code. This is a whole-word
        text search, so it will include the definition line itself
        alongside call sites — treat the result as "everywhere this name
        appears," not a precise semantic reference graph.
    """
    pattern = rf"\b{re.escape(name)}\b"
    return _grep(pattern, "*.py")


# ---------------------------------------------------------------------------
# Execution Tools (Section 4.5.3)
# ---------------------------------------------------------------------------

@mcp.tool()
def run_tests() -> str:
    """Execute the SWE-bench evaluation script for this task's testbed.

    The script content is read from the EVAL_SCRIPT environment variable;
    if that is unset, it falls back to a file named 'eval.sh' at the root
    of TESTBED_PATH (agent_swebench copies the task's eval_script there).

    Returns:
        JSON string: {"exit_code": int, "stdout": str, "stderr": str}
    """
    root = _testbed_path()
    script = os.environ.get("EVAL_SCRIPT")

    if not script:
        eval_file = root / "eval.sh"
        if eval_file.is_file():
            script = eval_file.read_text()
        else:
            return json.dumps({
                "exit_code": -1,
                "stdout": "",
                "stderr": (
                    "No eval script found: EVAL_SCRIPT env var is unset "
                    "and no eval.sh exists at TESTBED_PATH root."
                ),
            })

    # bash, not shell=True's /bin/sh: eval scripts use `set -o pipefail`.
    # stderr is merged in: the set -x markers and most test runners write there.
    code, out, err = _run(["bash", "-c", f"exec 2>&1\n{script}"], cwd=str(root), timeout=900)  # SWE-bench's 900s task timeout
    return json.dumps({"exit_code": code, "stdout": _test_section(out), "stderr": err})


def _test_section(output: str) -> str:
    """SWE-bench eval scripts wrap the test run in '>>>>> Start/End Test
    Output' markers, after pages of setup noise (locale-gen, git show,
    pip install). Return only the lines between them when present."""
    lines = output.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if ">>>>> Start Test Output" in line), None)
    if start is None:
        return output
    end = next((i for i in range(start + 1, len(lines)) if ">>>>> End Test Output" in lines[i]), len(lines))
    return "".join(lines[start + 1:end])


@mcp.tool()
def get_patch() -> str:
    """Retrieve the unified git diff of all changes made to the repository.

    Returns:
        The patch text from `git -c core.fileMode=false diff`, or an
        "Error: ..." message if the testbed is not a git repository or
        the command fails.
    """
    root = _testbed_path()
    code, out, err = _run(
        ["git", "-c", "core.fileMode=false", "diff"], cwd=str(root), timeout=30
    )
    if code != 0:
        return f"Error: git diff failed (exit {code}): {err}"
    return out


@mcp.tool()
def run_command(command: str, workdir: str = "") -> str:
    """Execute a shell command in a given working directory.

    Args:
        command: The shell command to execute.
        workdir: Absolute path, or a path relative to TESTBED_PATH.
            Defaults to the TESTBED_PATH root if not given.

    Returns:
        JSON string: {"stdout": str, "stderr": str, "exit_code": int}
    """
    cwd = _resolve(workdir) if workdir else _testbed_path()
    if not cwd.is_dir():
        return json.dumps({"stdout": "", "stderr": f"workdir not found: {workdir}", "exit_code": -1})

    code, out, err = _run(command, cwd=str(cwd))
    return json.dumps({"stdout": out, "stderr": err, "exit_code": code})


# ---------------------------------------------------------------------------
# Resources / Prompts (Section 4.2 — "tools, resources, and prompts must be exposed")
# ---------------------------------------------------------------------------

@mcp.resource("swebench://tool-docs")
def tool_docs() -> str:
    """Documentation of available SWE-bench tools, for LLM/manual generation."""
    return (
        "read_file(filepath, start_line=1, end_line=None) -> numbered file content\n"
        "edit_file(filepath, old_str, new_str) -> replace unique exact string\n"
        "list_files(directory='.', pattern='*') -> newline-separated file paths\n"
        "search_code(pattern, file_pattern='*.py') -> grep-like results\n"
        "search_function_or_class_definition_in_code(name) -> definition locations\n"
        "find_references(name, filepath='', line=0) -> usage locations\n"
        "run_tests() -> JSON {exit_code, stdout, stderr}\n"
        "get_patch() -> unified git diff of all changes\n"
        "run_command(command, workdir='') -> JSON {stdout, stderr, exit_code}"
    )


@mcp.prompt()
def solve_swebench_task(problem_statement: str) -> str:
    """Prompt template for solving a SWE-bench task."""
    return (
        f"Fix the following issue in the repository:\n{problem_statement}\n\n"
        "Explore the codebase with search_code / search_function_or_class_definition_in_code, "
        "read relevant files with read_file, make changes with edit_file, verify with run_tests, "
        "then call final_answer(get_patch()) once the fix is confirmed."
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport", choices=["stdio", "streamable-http"], default="stdio"
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", port=args.port)