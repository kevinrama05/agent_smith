"""
sandbox.py — Agent Smith's execution sandbox (Section 4.2 of the subject).

Covers:

  - Import allowlist            (authorized_imports, with `.*` submodule support)
  - Filesystem allowlist        (allowed_directories, path-traversal safe)
  - Execution timeout           (SIGALRM, POSIX)
  - Memory limit                (RLIMIT_AS, POSIX)
  - Restricted builtins         (dangerous names removed/overridden)
  - final_answer() construct    (NOT an MCP tool — always present)
  - Correct exception handling  (KeyboardInterrupt / SystemExit propagate)
  - Explicit, non-silent feedback for every failure mode Section 4.1 lists
  - MCP client integration      (stdio AND streamable HTTP), dynamic tool
                                 wrapper injection, and an auto-generated
                                 sandbox manual built from the connected
                                 server's tool schemas

Written against `mcp>=2` (see pyproject.toml): `mcp.client.stdio`,
`mcp.client.streamable_http.streamable_http_client`, `ClientSession`,
and snake_case `Tool.input_schema`.

CLI usage (per the subject):

    uv run sandbox                                  # interactive REPL
    uv run sandbox sandbox_template.json            # with custom config
    uv run sandbox --mcp-stdio "python mcp_tools_mbpp.py" sandbox_template.json
    uv run sandbox --mcp-server <URL>
"""

from __future__ import annotations

import argparse
import asyncio
import builtins as _builtins_module
import contextlib
import io
import json
import os
import shlex
import signal
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config (verbatim shape from Section 4.2 of the subject)
# ---------------------------------------------------------------------------


class SandboxConfig(BaseModel):
    """Sandbox configuration for student solutions.

    Uses allowlist approach: only imports in authorized_imports are allowed.
    Everything else is blocked by default.
    """

    authorized_imports: List[str] = Field(
        default_factory=lambda: [
            "math", "math.*",
            "collections", "collections.*",
            "itertools", "re", "json",
            "typing", "typing.*",
            "functools", "operator",
            "heapq", "bisect", "copy",
            "string", "random",
            "datetime", "datetime.*",
            "array", "cmath",
        ]
    )
    allowed_directories: List[str] = Field(
        default_factory=lambda: ["/testbed", "/tmp/agent"]
    )
    max_execution_time_seconds: int = 30
    max_memory_mb: int = 512


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """What the sandbox hands back to the agent loop after one Code step."""

    output: str                  # combined stdout/stderr/error text (the "Observation")
    is_final_answer: bool = False
    final_answer_value: Any = None
    truncated: bool = False


MAX_OUTPUT_CHARS = 8000  # Section 4.1: "Tool output was truncated due to size limits"


# ---------------------------------------------------------------------------
# Signals used for internal control flow (never leak these to the caller
# as ordinary errors — they're not failures, they're protocol)
# ---------------------------------------------------------------------------


class _FinalAnswerSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


class _SandboxTimeout(Exception):
    pass


# ---------------------------------------------------------------------------
# Import restriction
# ---------------------------------------------------------------------------


def _import_is_authorized(name: str, authorized: List[str]) -> bool:
    """`name` is what's being imported, e.g. 'collections.abc' or 'math'."""
    if name in authorized:
        return True
    # Wildcard entries: "collections.*" authorizes "collections" itself
    # and any of its submodules.
    for entry in authorized:
        if entry.endswith(".*"):
            base = entry[:-2]
            if name == base or name.startswith(base + "."):
                return True
    return False


def _make_restricted_import(authorized: List[str]):
    real_import = _builtins_module.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split(".")[0]
        # Check both the full dotted path and its top-level package, since
        # authorized_imports may list either ("math" or "collections.abc").
        if not (_import_is_authorized(name, authorized) or _import_is_authorized(top_level, authorized)):
            raise ImportError(
                f"Import of '{name}' is not allowed. "
                f"Authorized imports: {', '.join(sorted(authorized))}"
            )
        return real_import(name, globals, locals, fromlist, level)

    return restricted_import


# ---------------------------------------------------------------------------
# Filesystem restriction
# ---------------------------------------------------------------------------


class SandboxPermissionError(PermissionError):
    pass


def _make_restricted_open(allowed_directories: List[str]):
    real_open = _builtins_module.open
    resolved_allowed = [str(Path(d).resolve()) for d in allowed_directories]

    def restricted_open(file, mode="r", *args, **kwargs):
        # Resolve symlinks / ".." so traversal tricks can't escape the allowlist.
        try:
            target = str(Path(file).resolve())
        except Exception as e:
            raise SandboxPermissionError(f"Could not resolve path {file!r}: {e}")

        if not any(
            target == d or target.startswith(d + os.sep) for d in resolved_allowed
        ):
            raise SandboxPermissionError(
                f"Access to '{file}' is denied. "
                f"Allowed directories: {', '.join(allowed_directories)}"
            )
        return real_open(file, mode, *args, **kwargs)

    return restricted_open


# ---------------------------------------------------------------------------
# Restricted builtins
# ---------------------------------------------------------------------------

# Names deliberately excluded (privilege-escalation / sandbox-escape surface).
# Note: this is allowlist-of-exclusions on top of a broad safe set, not a true
# capability sandbox — see the known-limitations note at the bottom of this file.
_DANGEROUS_BUILTINS = {
    "eval", "exec", "compile", "input",
    "breakpoint", "help",
    "globals", "locals", "vars", "dir",
    "exit", "quit",
    "memoryview",
}


def _build_safe_builtins(authorized_imports: List[str], allowed_directories: List[str]) -> dict:
    safe = {
        name: getattr(_builtins_module, name)
        for name in dir(_builtins_module)
        if name not in _DANGEROUS_BUILTINS and not name.startswith("_")
    }
    safe["__import__"] = _make_restricted_import(authorized_imports)
    safe["open"] = _make_restricted_open(allowed_directories)
    return safe


# ---------------------------------------------------------------------------
# The Sandbox
# ---------------------------------------------------------------------------


class Sandbox:
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.namespace: dict = {}
        # Tool wrappers discovered from a connected MCP server get merged
        # into this dict before each execute() call, alongside final_answer
        # (which is NOT an MCP tool — injected separately, always present).
        self.extra_namespace: dict = {}
        self.mcp_connector: Optional["MCPToolConnector"] = None
        self._reset_namespace()

    def attach_mcp(self, connector: "MCPToolConnector") -> None:
        """Wire a connected MCPToolConnector's tools into the sandbox
        namespace and manual. Swapping connectors (a different MCP server)
        just means calling this again — the manual and callables update
        automatically."""
        self.mcp_connector = connector
        self.extra_namespace.update(connector.build_wrappers())

    def _reset_namespace(self) -> None:
        safe_builtins = _build_safe_builtins(
            self.config.authorized_imports, self.config.allowed_directories
        )
        self.namespace = {"__builtins__": safe_builtins}

    def _inject_final_answer(self) -> None:
        def final_answer(value: Any):
            raise _FinalAnswerSignal(value)

        self.namespace["final_answer"] = final_answer

    def _apply_limits(self):
        """POSIX-only: SIGALRM for timeout, RLIMIT_AS for memory.
        On platforms without these (e.g. Windows), limits are skipped with
        a warning rather than crashing — swap in a subprocess-based
        implementation there if you need real enforcement."""
        applied = []

        if hasattr(signal, "SIGALRM"):
            def _on_alarm(signum, frame):
                raise _SandboxTimeout()

            old_handler = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(self.config.max_execution_time_seconds)
            applied.append(("alarm", old_handler))

        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            new_limit = self.config.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
            applied.append(("rlimit", (soft, hard)))
        except (ImportError, ValueError, OSError):
            pass  # not available on this platform / already tighter

        return applied

    def _clear_limits(self, applied) -> None:
        for kind, old in applied:
            if kind == "alarm":
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
            elif kind == "rlimit":
                import resource

                resource.setrlimit(resource.RLIMIT_AS, old)

    def execute(self, code: str) -> ExecutionResult:
        """Run one Code step. Never raises for ordinary user-code errors —
        every failure mode becomes explicit text in ExecutionResult.output,
        per Section 4.1's "the LLM should never be left guessing" rule.
        KeyboardInterrupt / SystemExit are the one exception: they propagate,
        per Section 4.2's exception-propagation requirement.
        """
        if not code or not code.strip():
            return ExecutionResult(output="No code was provided to execute.")

        try:
            compiled = compile(code, "<agent_step>", "exec")
        except SyntaxError as e:
            return ExecutionResult(
                output=f"SyntaxError: {e.msg} (line {e.lineno}, offset {e.offset}). "
                       f"The code was not executed."
            )

        # Fresh namespace each step but reuse persistent state across
        # steps by keeping self.namespace alive between calls (variables
        # the LLM defines in step N are visible in step N+1).
        self.namespace.setdefault("__builtins__", _build_safe_builtins(
            self.config.authorized_imports, self.config.allowed_directories
        ))
        self._inject_final_answer()
        self.namespace.update(self.extra_namespace)  # MCP tool wrappers, once wired

        stdout_buf = io.StringIO()
        applied_limits = self._apply_limits()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
                exec(compiled, self.namespace)
        except _FinalAnswerSignal as sig:
            return ExecutionResult(
                output=stdout_buf.getvalue() or "(final_answer called)",
                is_final_answer=True,
                final_answer_value=sig.value,
            )
        except _SandboxTimeout:
            partial = stdout_buf.getvalue()
            return ExecutionResult(
                output=(
                    f"Execution timed out after {self.config.max_execution_time_seconds}s. "
                    f"Partial output before timeout:\n{partial}"
                ),
                truncated=True,
            )
        except (KeyboardInterrupt, SystemExit):
            raise  # must reach the agent loop, per Section 4.2
        except MemoryError:
            return ExecutionResult(
                output=f"MemoryError: execution exceeded the {self.config.max_memory_mb}MB limit."
            )
        except Exception:
            tb_text = traceback.format_exc()
            return self._finalize_output(stdout_buf.getvalue() + tb_text)
        finally:
            self._clear_limits(applied_limits)

        return self._finalize_output(stdout_buf.getvalue() or "(no output)")

    @staticmethod
    def _finalize_output(text: str) -> ExecutionResult:
        if len(text) > MAX_OUTPUT_CHARS:
            return ExecutionResult(
                output=text[:MAX_OUTPUT_CHARS]
                + f"\n... [truncated: output exceeded {MAX_OUTPUT_CHARS} characters]",
                truncated=True,
            )
        return ExecutionResult(output=text)

    def manual_text(self) -> str:
        """Human/LLM-readable description of what's callable right now.
        Feed this straight into your agent's system prompt."""
        lines = [
            "Available in this sandbox:",
            "  final_answer(value) -- call this with your solution to finish the task.",
        ]
        if self.mcp_connector is not None:
            lines.append("")
            lines.append("MCP tools (from the connected server):")
            lines.append(self.mcp_connector.manual_text())
        lines += [
            "",
            f"Authorized imports: {', '.join(self.config.authorized_imports)}",
            f"Allowed directories: {', '.join(self.config.allowed_directories)}",
            f"Execution timeout: {self.config.max_execution_time_seconds}s, "
            f"memory limit: {self.config.max_memory_mb}MB",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP client integration
#
# The sandbox's exec() is synchronous, but the MCP client is async. We
# bridge this with a dedicated background thread running its own asyncio
# event loop: the connection is opened once and kept alive there for the
# whole sandbox session, and every tool call is scheduled onto that loop
# via run_coroutine_threadsafe() and blocked on from the sandboxed code's
# (synchronous) point of view.
#
# This project is pinned to mcp>=2 (see pyproject.toml): `streamable_http_client`
# yields a (read, write) pair, and Tool fields are snake_case
# (`tool.input_schema`, not the old v1 `tool.inputSchema`).
# ---------------------------------------------------------------------------


class MCPConnectionError(RuntimeError):
    pass


class MCPToolConnector:
    """Connects to one MCP server (stdio or streamable HTTP), discovers its
    tools, and exposes them as plain synchronous Python callables."""

    def __init__(self, mode: str, target: str, extra_env: Optional[dict] = None):
        if mode not in ("stdio", "http"):
            raise ValueError(f"mode must be 'stdio' or 'http', got {mode!r}")
        self.mode = mode
        self.target = target
        # stdio only: merged on top of the current process's environment
        # when launching the server subprocess -- e.g. {"TESTBED_PATH": "/testbed"}
        # for local (non-Docker) SWE-bench testing.
        self.extra_env = extra_env or {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._exit_stack = None
        self.tools_info: list = []

    # -- lifecycle -----------------------------------------------------

    def connect(self, timeout: float = 30.0) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._async_connect(), self._loop)
        try:
            future.result(timeout=timeout)
        except Exception as e:
            self.close()
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def _async_connect(self) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession

        self._exit_stack = AsyncExitStack()

        if self.mode == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            parts = shlex.split(self.target)
            if not parts:
                raise ValueError("Empty --mcp-stdio command.")
            merged_env = {**os.environ, **self.extra_env} if self.extra_env else None
            params = StdioServerParameters(command=parts[0], args=parts[1:], env=merged_env)
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        else:
            # This project is pinned to mcp>=2 (see pyproject.toml), whose
            # streamable_http_client yields a (read, write) pair -- the v1
            # client (streamablehttp_client, a 3-tuple including a session-id
            # getter) is a different, older API and is not what's installed.
            from mcp.client.streamable_http import streamable_http_client

            read, write = await self._exit_stack.enter_async_context(
                streamable_http_client(self.target)
            )

        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._session = session

        tools_result = await session.list_tools()
        self.tools_info = list(tools_result.tools)

    def close(self) -> None:
        if self._loop and self._exit_stack:
            future = asyncio.run_coroutine_threadsafe(self._exit_stack.aclose(), self._loop)
            try:
                future.result(timeout=10)
            except Exception:
                pass  # best-effort cleanup; don't let teardown crash the REPL
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    # -- tool calling ----------------------------------------------------

    def _call_tool_sync(self, name: str, kwargs: dict, timeout: float = 120.0) -> str:
        if self._session is None:
            raise MCPConnectionError("MCP session is not connected.")
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, kwargs), self._loop
        )
        result = future.result(timeout=timeout)

        # result.content is a list of content blocks (usually TextContent).
        # Flatten to plain text for the sandboxed code to print/inspect.
        parts = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else str(block))
        text = "\n".join(parts)

        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            return f"Error from tool '{name}': {text}"
        return text

    def build_wrappers(self) -> dict:
        """One synchronous Python callable per discovered MCP tool, matching
        the param order in its schema so both positional and keyword calls
        work, e.g. read_file("models.py", 1, 50) or read_file(filepath=...)."""
        wrappers = {}
        for tool in self.tools_info:
            wrappers[tool.name] = self._make_wrapper(tool)
        return wrappers

    def _make_wrapper(self, tool):
        schema = getattr(tool, "input_schema", None) or {}
        param_names = list((schema.get("properties") or {}).keys())

        def wrapper(*args, **kwargs):
            merged = dict(zip(param_names, args))
            merged.update(kwargs)
            text = self._call_tool_sync(tool.name, merged)
            # Runs inside the sandbox's redirected stdout, so this lands in
            # the Observation: a failed edit_file(...) whose result is never
            # printed would otherwise look like it succeeded.
            if text.startswith("Error"):
                print(f"[{tool.name}] {text}")
            return text

        wrapper.__name__ = tool.name
        wrapper.__doc__ = tool.description or f"MCP tool '{tool.name}'."
        return wrapper

    # -- manual generation -------------------------------------------------

    def manual_text(self) -> str:
        """Dynamically generated from the connected server's tool schemas,
        per Section 4.2 point 5 — this is what changes automatically when a
        different MCP server is connected."""
        if not self.tools_info:
            return "(no tools discovered)"

        lines = []
        for tool in self.tools_info:
            schema = getattr(tool, "input_schema", None) or {}
            props = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            params = ", ".join(
                f"{name}{'' if name in required else '=...'}: {info.get('type', 'any')}"
                for name, info in props.items()
            )
            desc = tool.description or "(no description)"
            lines.append(f"  {tool.name}({params}) -- {desc}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI / interactive REPL
# ---------------------------------------------------------------------------


def _load_config(path: str | None) -> SandboxConfig:
    if not path:
        return SandboxConfig()
    with open(path) as f:
        data = json.load(f)
    return SandboxConfig(**data)


def _run_repl(sandbox: Sandbox) -> None:
    print("Agent Smith interactive sandbox. Type 'exit' or Ctrl-D to quit.")
    print(sandbox.manual_text())
    print()
    while True:
        try:
            line = input(">>> ")
        except EOFError:
            print()
            break
        if line.strip() == "exit":
            break
        result = sandbox.execute(line)
        print(result.output)
        if result.is_final_answer:
            print(f"[final_answer received]: {result.final_answer_value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sandbox")
    parser.add_argument("config", nargs="?", help="Path to a SandboxConfig JSON file.")
    parser.add_argument("--mcp-stdio", help="Command to launch an MCP server over stdio.")
    parser.add_argument("--mcp-server", help="URL of an MCP server over streamable HTTP.")
    args = parser.parse_args()

    if args.mcp_stdio and args.mcp_server:
        parser.error("Use --mcp-stdio OR --mcp-server, not both.")

    config = _load_config(args.config)
    sandbox = Sandbox(config)

    connector: Optional[MCPToolConnector] = None
    try:
        if args.mcp_stdio:
            connector = MCPToolConnector("stdio", args.mcp_stdio)
        elif args.mcp_server:
            connector = MCPToolConnector("http", args.mcp_server)

        if connector is not None:
            print(f"[sandbox] Connecting to MCP server ({connector.mode})...", file=sys.stderr)
            connector.connect()
            sandbox.attach_mcp(connector)
            tool_names = ", ".join(t.name for t in connector.tools_info) or "(none)"
            print(f"[sandbox] Connected. Discovered tools: {tool_names}", file=sys.stderr)

        _run_repl(sandbox)
    finally:
        if connector is not None:
            connector.close()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Known limitations / design notes:
#
# 1. This is an exclusion list on top of real Python builtins, not a true
#    capability sandbox. A determined payload could still reach dangerous
#    functionality via attribute-chasing (e.g. crawling __subclasses__() off
#    an allowed object). Fully closing this would need AST-level rejection of
#    such patterns or running untrusted code in a separate OS process.
# 2. Timeout and memory enforcement are POSIX-only (SIGALRM + RLIMIT_AS).
# 3. Network access is blocked indirectly: `socket` (and every module that
#    wraps it) is outside the import allowlist, so sandboxed code cannot open
#    connections unless a config explicitly authorizes such a module.
# 4. MCP tool calls run outside every sandbox restriction above (import,
#    filesystem, timeout, memory), matching the subject's framing that "the
#    sandbox and MCP tools are independent security domains". A slow tool
#    call is bounded by _call_tool_sync's own 120s timeout instead.
# 5. Keep the MCP servers on the same mcp major version as this client
#    (mcp>=2); mixing v1 and v2 fails at the protocol level.
# ---------------------------------------------------------------------------
