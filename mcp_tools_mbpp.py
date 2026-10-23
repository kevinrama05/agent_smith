"""
mcp_tools_mbpp.py — MCP server exposing the MBPP run_tests tool (Section 4.3).

    python mcp_tools_mbpp.py [--transport stdio|streamable-http] [--port 8000]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("mbpp-tools")


@mcp.tool()
def run_tests(code: str, test_list: list[str]) -> str:
    """Run a candidate Python solution against MBPP test assertions.

    Args:
        code: The full Python source of the candidate function(s).
        test_list: A list of `assert` statements to validate the solution.

    Returns:
        JSON string: {"success": bool, "output": str}
    """
    script = code + "\n\n" + "\n".join(test_list) + "\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(script)
        path = f.name

    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        success = result.returncode == 0
        output = (result.stdout + result.stderr).strip()
        if success and not output:
            output = "All test assertions passed."
    except subprocess.TimeoutExpired:
        success = False
        output = "Execution timed out after 10 seconds."
    finally:
        os.unlink(path)

    return json.dumps({"success": success, "output": output})


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
    