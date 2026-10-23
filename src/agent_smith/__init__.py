"""agent_smith package.

uv_build only packages this `src/agent_smith` directory into the installed
project, while the actual implementation (sandbox.py, agent_loop.py, the
agent CLIs, etc.) lives at the repository root so it can be imported with
plain `from sandbox import ...` style imports (as required by
`uv run python -m agent_mbpp` / `agent_swebench`). Console scripts declared
in [project.scripts], however, run from the *installed* package, which
can't see the root-level modules unless we point it there explicitly --
hence the small sys.path bootstrap below, which works because this project
is always run locally via `uv run` (an editable install), never installed
as a standalone distributable artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    """Entry point for `uv run agent-smith`."""
    print(
        "Agent Smith -- see README.md for the actual CLIs:\n"
        "  uv run sandbox\n"
        "  uv run python -m agent_mbpp --task-file ... --output ... "
        "--model-name ... --provider-url ...\n"
        "  uv run python -m agent_swebench --task-file ... --output ... "
        "--model-name ... --provider-url ..."
    )


def sandbox_main() -> None:
    """Entry point for `uv run sandbox` (Section 4.2 of the subject)."""
    from sandbox import main as _sandbox_main

    _sandbox_main()
