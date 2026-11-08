"""Tool registry and sandboxed file tools. This is the core of chapter b02.

Idea: adding a tool means adding ONE handler. The registry does the rest:
it produces the toolSpec the model sees, validates arguments, runs the handler,
and turns every failure into an error result the model can read and react to,
instead of crashing the loop.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import lab_common as lc

MAX_FILE_CHARS = 20_000
MAX_LIST_ENTRIES = 200


@dataclass
class Tool:
    name: str
    description: str
    properties: dict
    required: list
    handler: Callable[[dict], Any]
    risk: str = "read"  # "read" or "write": chapter b03 uses this to decide what needs approval


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        properties: dict,
        required: list,
        handler: Callable[[dict], Any],
        risk: str = "read",
    ) -> "ToolRegistry":
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = Tool(name, description, properties, required, handler, risk)
        return self

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        return [lc.tool_spec(t.name, t.description, t.properties, t.required) for t in self._tools.values()]

    def tool_config(self) -> dict:
        return {"tools": self.specs()}

    def subset(self, names: list[str]) -> "ToolRegistry":
        """A new registry holding only the named tools (used for subagents)."""
        sub = ToolRegistry()
        for name in names:
            tool = self._tools[name]
            sub._tools[name] = tool
        return sub

    def call(self, name: str, tool_input: dict) -> tuple[Any, bool]:
        """Run a tool. Returns (output, is_error). Never raises."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool '{name}'. Available tools: {', '.join(self.names())}", True
        if not isinstance(tool_input, dict):
            return "Tool input must be a JSON object", True
        missing = [key for key in tool.required if key not in tool_input]
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}", True
        try:
            return tool.handler(tool_input), False
        except Exception as exc:  # noqa: BLE001 - the model should see every failure
            return f"{type(exc).__name__}: {exc}", True


# --------------------------------------------------------------------------
# Sandbox: file tools may only touch files under one folder
# --------------------------------------------------------------------------
class SandboxError(ValueError):
    pass


def resolve_in_sandbox(workdir: Path, relative: str) -> Path:
    """Resolve a path and refuse anything that escapes the sandbox folder."""
    root = workdir.resolve()
    candidate = (root / (relative or ".")).resolve()
    if candidate != root and root not in candidate.parents:
        raise SandboxError("path escapes the sandbox folder")
    return candidate


def default_workdir() -> Path:
    """The folder the file tools may touch. Override with LAB_WORKDIR."""
    workdir = Path(os.getenv("LAB_WORKDIR") or (lc.ROOT / "workspace"))
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def build_default_registry(workdir: Path | None = None) -> ToolRegistry:
    """calculate, list_dir and read_text_file: three safe, read-only tools."""
    root = (workdir or default_workdir()).resolve()

    def calculate(args: dict) -> str:
        return lc.safe_calculate(str(args["expression"]))

    def list_dir(args: dict) -> str:
        target = resolve_in_sandbox(root, str(args.get("path", ".")))
        if not target.is_dir():
            raise NotADirectoryError(f"not a folder: {args.get('path', '.')}")
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [p.name + ("/" if p.is_dir() else "") for p in entries[:MAX_LIST_ENTRIES]]
        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"... {len(entries) - MAX_LIST_ENTRIES} more entries not shown")
        return "\n".join(lines) or "(empty folder)"

    def read_text_file(args: dict) -> str:
        target = resolve_in_sandbox(root, str(args["path"]))
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {args['path']}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_FILE_CHARS:
            return text[:MAX_FILE_CHARS] + f"\n... [truncated: {len(text) - MAX_FILE_CHARS} more characters]"
        return text

    registry = ToolRegistry()
    registry.register(
        "calculate",
        "Evaluate an arithmetic expression such as '17 * 23'. Supports + - * / // % ** and parentheses.",
        {"expression": {"type": "string", "description": "The arithmetic expression"}},
        ["expression"],
        calculate,
    )
    registry.register(
        "list_dir",
        "List files and folders inside the workspace. Folders end with '/'.",
        {"path": {"type": "string", "description": "Folder relative to the workspace root. Default '.'"}},
        [],
        list_dir,
    )
    registry.register(
        "read_text_file",
        "Read a UTF-8 text file from the workspace (long files are truncated).",
        {"path": {"type": "string", "description": "File path relative to the workspace root"}},
        ["path"],
        read_text_file,
    )
    return registry
