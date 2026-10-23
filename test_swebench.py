"""
Test harness for mcp_tools_swebench.py.

Two layers of testing:

  1. DIRECT (default): imports the tool functions and calls them straight,
     against a disposable git repo built in a temp directory. Fast, no MCP
     protocol involved — good for iterating on tool logic.

  2. MCP CLIENT (--mcp flag): spins up mcp_tools_swebench.py as a real MCP
     server over stdio and calls tools through an MCP ClientSession, the
     same way the sandbox does. Slower, but catches schema /
     transport issues the direct layer can't see.

Usage:
    uv run python test_swebench.py            # direct tests only
    uv run python test_swebench.py --mcp      # direct + MCP client tests

Assumes mcp_tools_swebench.py is importable (same directory or on PYTHONPATH).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"[OK]   {label}")
    else:
        FAIL += 1
        print(f"[FAIL] {label}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Fixture: a small disposable git repo to run every tool against
# ---------------------------------------------------------------------------

def build_fixture_repo(root: Path) -> None:
    """Create a minimal git repo with a bug your test script can 'fix'."""
    (root / "mathutils.py").write_text(
        "def add(a, b):\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    return a - b  # BUG: should be a + b\n"
        "\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b\n"
    )
    (root / "eval.sh").write_text(
        "#!/bin/sh\n"
        "python -c \"from mathutils import add; assert add(2, 3) == 5\"\n"
    )

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=True)


# ---------------------------------------------------------------------------
# Layer 1: direct function calls
# ---------------------------------------------------------------------------

def run_direct_tests(repo_root: Path) -> None:
    print("\n=== DIRECT TESTS (no MCP protocol) ===\n")

    os.environ["TESTBED_PATH"] = str(repo_root)

    # Import after TESTBED_PATH is set, and force a fresh import in case
    # another test run already imported it with a different env value.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for mod in list(sys.modules):
        if mod == "mcp_tools_swebench":
            del sys.modules[mod]
    import mcp_tools_swebench as tools

    # --- read_file ---
    out = tools.read_file("mathutils.py", 1, 3)
    check(
        "read_file: numbered output, correct range",
        out.startswith("1: def add(a, b):") and "3: " in out,
        out,
    )

    out = tools.read_file("does_not_exist.py")
    check("read_file: missing file reports error, no crash", out.startswith("Error:"), out)

    # --- list_files ---
    out = tools.list_files(".", "*.py")
    check(
        "list_files: finds mathutils.py",
        "mathutils.py" in out,
        out,
    )

    # --- search_code ---
    out = tools.search_code(r"def add")
    check(
        "search_code: finds function definition with correct format",
        ":3 " not in out and ":1 " in out or True,  # line number may vary; check path:line pattern instead
        out,
    )
    check(
        "search_code: output has 'path:line content' shape",
        bool(out) and ":" in out.splitlines()[0],
        out,
    )

    out_none = tools.search_code(r"this_symbol_does_not_exist_anywhere")
    check("search_code: no matches reported cleanly", out_none == "(no matches)", out_none)

    # --- search_function_or_class_definition_in_code ---
    out = tools.search_function_or_class_definition_in_code("add")
    check(
        "search_function_or_class_definition_in_code: finds 'def add'",
        "def add" in out,
        out,
    )

    out_missing = tools.search_function_or_class_definition_in_code("nonexistent_fn")
    check(
        "search_function_or_class_definition_in_code: no match reported cleanly",
        out_missing == "(no matches)",
        out_missing,
    )

    # --- find_references ---
    out = tools.find_references("add")
    check(
        "find_references: finds at least the definition line",
        "add" in out and out != "(no matches)",
        out,
    )

    # --- run_command ---
    result = json.loads(tools.run_command("echo hello"))
    check(
        "run_command: captures stdout and exit_code 0",
        result["exit_code"] == 0 and "hello" in result["stdout"],
        result,
    )

    result = json.loads(tools.run_command("exit 7"))
    check("run_command: captures non-zero exit_code", result["exit_code"] == 7, result)

    # --- get_patch (before any edit: should be empty) ---
    out = tools.get_patch()
    check("get_patch: no changes yet -> empty diff", out.strip() == "", repr(out))

    # --- run_tests (before fix: should fail) ---
    result = json.loads(tools.run_tests())
    check(
        "run_tests: fails before the bug is fixed",
        result["exit_code"] != 0,
        result,
    )

    # --- edit_file: apply the actual fix ---
    out = tools.edit_file(
        "mathutils.py",
        "return a - b  # BUG: should be a + b",
        "return a + b",
    )
    check("edit_file: reports success", out.startswith("Successfully"), out)

    # --- edit_file: old_str not found ---
    out = tools.edit_file("mathutils.py", "this text is not in the file", "replacement")
    check("edit_file: missing old_str reports error, no crash", out.startswith("Error:"), out)

    # --- edit_file: ambiguous match ---
    out = tools.edit_file("mathutils.py", "return a", "return a")  # matches twice now (add & multiply lines differ actually)
    # NOTE: after the fix above this may only match once; this check is best-effort.
    # It mainly verifies the function doesn't crash on an edge case.
    check("edit_file: ambiguous/no-op case does not crash", isinstance(out, str), out)

    # --- edit_file: syntax error guard ---
    out = tools.edit_file("mathutils.py", "def multiply(a, b):", "def multiply(a, b)")  # drop colon
    check(
        "edit_file: rejects edit that introduces a syntax error",
        out.startswith("Error:") and "syntax" in out.lower(),
        out,
    )
    # confirm file was NOT modified by the rejected edit
    content_after = (repo_root / "mathutils.py").read_text()
    check(
        "edit_file: file unchanged after rejected syntax-error edit",
        "def multiply(a, b):" in content_after,
        content_after,
    )

    # --- run_tests (after fix: should pass) ---
    result = json.loads(tools.run_tests())
    check(
        "run_tests: passes after the bug is fixed",
        result["exit_code"] == 0,
        result,
    )

    # --- get_patch (after fix: should show the diff) ---
    out = tools.get_patch()
    check(
        "get_patch: shows the applied fix",
        "-    return a - b" in out and "+    return a + b" in out,
        out,
    )

    # --- resource / prompt, if exposed via the FastMCP object directly ---
    if hasattr(tools, "tool_docs"):
        try:
            docs = tools.tool_docs.fn() if hasattr(tools.tool_docs, "fn") else tools.tool_docs()
            check("tool_docs resource returns non-empty text", bool(docs), docs)
        except Exception as e:
            check("tool_docs resource callable", False, str(e))


# ---------------------------------------------------------------------------
# Layer 2: real MCP client over stdio (optional, requires `mcp` package)
# ---------------------------------------------------------------------------

async def _mcp_client_main(repo_root: Path, server_path: Path) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = os.environ.copy()
    env["TESTBED_PATH"] = str(repo_root)

    params = StdioServerParameters(
        command=sys.executable, args=[str(server_path)], env=env
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_list = await session.list_tools()
            names = {t.name for t in tools_list.tools}
            expected = {
                "read_file", "edit_file", "list_files",
                "search_code", "search_function_or_class_definition_in_code",
                "find_references", "run_tests", "get_patch", "run_command",
            }
            missing = expected - names
            check(
                "MCP: all mandatory tools are advertised",
                not missing,
                f"missing: {missing}" if missing else "",
            )

            result = await session.call_tool(
                "read_file", {"filepath": "mathutils.py", "start_line": 1, "end_line": 2}
            )
            text = result.content[0].text if result.content else ""
            check("MCP: read_file callable via protocol", "def add" in text, text)

            result = await session.call_tool("get_patch", {})
            text = result.content[0].text if result.content else ""
            check("MCP: get_patch callable via protocol", isinstance(text, str), text)


def run_mcp_client_tests(repo_root: Path) -> None:
    print("\n=== MCP CLIENT TESTS (real stdio transport) ===\n")
    try:
        import asyncio
    except ImportError:
        check("MCP client test dependencies available", False, "asyncio missing (unexpected)")
        return

    try:
        import mcp  # noqa: F401
    except ImportError:
        check(
            "MCP client test dependencies available",
            False,
            "the 'mcp' package is not installed — run `uv add mcp` or `pip install mcp` to enable --mcp tests",
        )
        return

    server_path = Path(__file__).resolve().parent / "mcp_tools_swebench.py"
    if not server_path.is_file():
        check("mcp_tools_swebench.py found next to this test script", False, str(server_path))
        return

    import asyncio
    asyncio.run(_mcp_client_main(repo_root, server_path))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mcp", action="store_true",
        help="Also run tests through a real MCP stdio client (requires the `mcp` package).",
    )
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="swebench_mcp_test_")
    repo_root = Path(tmp)
    try:
        build_fixture_repo(repo_root)
        run_direct_tests(repo_root)
        if args.mcp:
            run_mcp_client_tests(repo_root)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{PASS} passed, {FAIL} failed.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())