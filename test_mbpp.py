"""Direct unit checks for mcp_tools_mbpp.run_tests.

    uv run python test_mbpp.py
"""

from mcp_tools_mbpp import run_tests
import json

def check(label, code, test_list, expect_success):
    result = json.loads(run_tests(code=code, test_list=test_list))
    status = "OK" if result["success"] == expect_success else "MISMATCH"
    print(f"[{label}] expected success={expect_success}, got={result['success']} -> {status}")
    print(f"    output: {result['output']!r}\n")
    assert result["success"] == expect_success, f"{label} failed expectation"

check(
    "Case 1: correct solution",
    code="def add(a, b):\n    return a + b",
    test_list=["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
    expect_success=True,
)

check(
    "Case 2: wrong solution",
    code="def add(a, b):\n    return a - b",
    test_list=["assert add(2, 3) == 5"],
    expect_success=False,
)

check(
    "Case 3: infinite loop",
    code="def bad():\n    while True: pass",
    test_list=["assert bad() is None"],
    expect_success=False,
)

print("All cases behaved as expected.")