"""
code_extraction.py — Agent Smith's format-agnostic code extraction layer
(Section 4.1, point 2).

Sits between the LLM provider and the sandbox:

    LLMResponse.text -> extract_code() -> ExtractionResult.code -> Sandbox.execute()

Tries formats in priority order and returns the FIRST match:

  1. Python code blocks (primary)  -- ```python ... ```
  2. XML tool calls (Anthropic-style) -- <invoke name="..."><parameter name="...">...
  3. JSON/Hermes tool calls -- <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  4. ReAct format -- Action: tool_name / Action Input: {...}

Non-Python formats (2-4) are converted into an equivalent Python function
call string, e.g. `result = read_file(filepath="/testbed/file.py")`, so the
sandbox stays completely format-agnostic -- it only ever sees Python.

Also implements the "never leave the LLM guessing" requirement: every
malformed-but-recoverable case (missing closing fence, untagged fence,
broken JSON) comes back with a `note` explaining exactly what was
tolerated, and the "nothing found at all" case comes back with `found=False`
and a clear message -- both meant to be fed back to the LLM as part of the
next Observation, not just logged internally.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    code: str                      # Python source to hand to Sandbox.execute(); "" if not found
    found: bool                    # False -> nothing executable was found in the response
    source_format: str             # "python_block" | "xml_invoke" | "json_tool_call" | "react" | "none"
    note: Optional[str] = None     # explanation for a malformed-but-recovered case, or the
                                    # "nothing found" message -- feed this to the LLM either way


# Markers a model might still emit even with stop_sequences configured
# (some providers echo the stop token before truncating, or the model
# free-runs past where it should have stopped). Stripped defensively from
# the end of extracted code.
_TRAILING_MARKERS = ("<end_code>", "</tool_call>", "</invoke>")


def _strip_trailing_markers(code: str) -> str:
    stripped = code.rstrip()
    changed = True
    while changed:
        changed = False
        for marker in _TRAILING_MARKERS:
            if stripped.endswith(marker):
                stripped = stripped[: -len(marker)].rstrip()
                changed = True
    return stripped


def _to_python_literal(raw: str) -> str:
    """Convert a raw string value (from an XML parameter, JSON arg, or
    ReAct Action Input) into Python source text for that literal.
    Tries JSON first (handles numbers/bools/null/nested structures
    correctly), falls back to treating it as a plain string."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return repr(raw)
    return repr(parsed)


# ---------------------------------------------------------------------------
# 1. Python code blocks (primary)
# ---------------------------------------------------------------------------


def _extract_python_block(text: str) -> Optional[ExtractionResult]:
    # Primary case: properly fenced and tagged.
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        code = _strip_trailing_markers(m.group(1))
        return ExtractionResult(code=code, found=True, source_format="python_block")

    # Malformed but recoverable: fenced, but missing the `python` tag.
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        code = _strip_trailing_markers(m.group(1))
        return ExtractionResult(
            code=code,
            found=True,
            source_format="python_block",
            note=(
                "The code block used a bare ``` fence without a 'python' "
                "language tag. It was interpreted as Python anyway."
            ),
        )

    # Malformed but recoverable: opening fence present, closing fence missing
    # (model got cut off, or forgot to close it). Take everything after the
    # opening fence as the code.
    m = re.search(r"```(?:python)?\s*\n(.*)\Z", text, re.DOTALL)
    if m:
        code = _strip_trailing_markers(m.group(1))
        if code:
            return ExtractionResult(
                code=code,
                found=True,
                source_format="python_block",
                note=(
                    "The code block was missing its closing ``` fence. "
                    "Everything after the opening fence was interpreted as Python."
                ),
            )

    return None


# ---------------------------------------------------------------------------
# 2. XML tool calls (Anthropic-style <invoke>)
# ---------------------------------------------------------------------------


def _extract_xml_invoke(text: str) -> Optional[ExtractionResult]:
    invokes = re.findall(
        r"<invoke\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</invoke>", text, re.DOTALL
    )
    if not invokes:
        return None

    statements = []
    for name, body in invokes:
        params = re.findall(
            r"<parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</parameter>", body, re.DOTALL
        )
        kwargs = ", ".join(f"{pname}={_to_python_literal(pval)}" for pname, pval in params)
        statements.append(f"result = {name}({kwargs})\nprint(result)")

    code = "\n".join(statements)
    return ExtractionResult(
        code=code,
        found=True,
        source_format="xml_invoke",
        note="Converted an Anthropic-style XML <invoke> tool call to an equivalent Python function call.",
    )


# ---------------------------------------------------------------------------
# 3. JSON / Hermes tool calls
# ---------------------------------------------------------------------------


def _extract_json_tool_call(text: str) -> Optional[ExtractionResult]:
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None

    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return ExtractionResult(
            code="",
            found=False,
            source_format="json_tool_call",
            note=f"Found a <tool_call> block, but its contents were not valid JSON: {e}",
        )

    name = data.get("name")
    if not name:
        return ExtractionResult(
            code="",
            found=False,
            source_format="json_tool_call",
            note="Found a <tool_call> block, but it had no 'name' field.",
        )

    args = data.get("arguments", {}) or {}
    kwargs = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    code = f"result = {name}({kwargs})\nprint(result)"
    return ExtractionResult(
        code=code,
        found=True,
        source_format="json_tool_call",
        note="Converted a JSON/Hermes-style <tool_call> block to an equivalent Python function call.",
    )


# ---------------------------------------------------------------------------
# 4. ReAct format
# ---------------------------------------------------------------------------


def _extract_react(text: str) -> Optional[ExtractionResult]:
    m = re.search(
        r"Action:\s*(\S+)\s*\n+Action Input:\s*(.+?)(?:\n\s*\n|\nObservation:|\Z)",
        text,
        re.DOTALL,
    )
    if not m:
        return None

    tool_name = m.group(1).strip()
    raw_input = m.group(2).strip()

    try:
        parsed = json.loads(raw_input)
    except (json.JSONDecodeError, ValueError):
        parsed = raw_input  # not JSON -- treat the whole thing as one string argument

    if isinstance(parsed, dict):
        kwargs = ", ".join(f"{k}={repr(v)}" for k, v in parsed.items())
        code = f"result = {tool_name}({kwargs})"
    else:
        code = f"result = {tool_name}({repr(parsed)})"
    code += "\nprint(result)"

    return ExtractionResult(
        code=code,
        found=True,
        source_format="react",
        note="Converted a ReAct-style Action/Action Input pair to an equivalent Python function call.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_PARSERS = (_extract_python_block, _extract_xml_invoke, _extract_json_tool_call, _extract_react)


def extract_code(llm_output: str) -> ExtractionResult:
    """Try each supported format in priority order; return the first match.

    If NOTHING matches, returns found=False with a message meant to be fed
    straight back to the LLM as the next Observation -- per Section 4.1,
    silent failure here is exactly what causes hallucinated observations
    and wasted iterations.
    """
    if not llm_output or not llm_output.strip():
        return ExtractionResult(
            code="",
            found=False,
            source_format="none",
            note="The model's response was empty. No code or tool call was found.",
        )

    for parser in _PARSERS:
        result = parser(llm_output)
        if result is not None:
            return result

    return ExtractionResult(
        code="",
        found=False,
        source_format="none",
        note=(
            "No valid Python code block, <invoke> tag, <tool_call> block, or "
            "ReAct Action/Action Input pair was found in the model's response. "
            "Wrap your code in a ```python ... ``` block."
        ),
    )


def feedback_text(result: ExtractionResult) -> str:
    """The text to prepend to (or stand in for) the sandbox Observation for
    this step. Empty string if there's nothing noteworthy to say."""
    if not result.found:
        return result.note or "No executable code was found in the model's response."
    if result.note:
        return f"[Note: {result.note}]"
    return ""