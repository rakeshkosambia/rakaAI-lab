"""b03: Permissions.

Motto: set the boundaries first, then grant freedom.

Two layers, because "the model might ask for it" must never mean "it happens":

  Layer 1 (in code):  PermissionPolicy decides allow / ask / deny BEFORE a tool runs.
  Layer 2 (in AWS):   an IAM policy limits what the credentials could do even if
                      layer 1 had a bug. See iam/ and the chapter README.

A denied call is not a crash: the model receives an error result that says why,
and can choose another approach.

Run:  python b03_permission/code.py --offline
      python b03_permission/code.py --mode readonly --prompt "Save a note called todo.txt"
Modes: readonly (no writes), default (writes need your approval), auto (writes inside the sandbox allowed)
"""
from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab_common as lc  # noqa: E402
import lab_tools  # noqa: E402

DEFAULT_PROMPT = "Write a short file called summary.txt with the text 'Orders upgrade is in progress'."
SYSTEM = [{"text": "You are a careful assistant with file tools. If a tool is refused, explain and suggest an alternative. Keep answers short."}]

# File names the agent may never read or write, even inside the sandbox.
DENY_PATTERNS = [".env", ".env.*", "*.pem", "*.key", "*credentials*", "*secret*", "id_rsa*"]


class PermissionPolicy:
    """Decide allow / ask / deny for each tool call. Deny always beats allow."""

    def __init__(self, mode: str = "default", deny_patterns: list[str] | None = None):
        if mode not in ("readonly", "default", "auto"):
            raise ValueError("mode must be readonly, default or auto")
        self.mode = mode
        self.deny_patterns = deny_patterns if deny_patterns is not None else DENY_PATTERNS

    def check(self, tool, tool_input: dict) -> tuple[str, str]:
        """Return (decision, reason). decision is 'allow', 'ask' or 'deny'."""
        path = str(tool_input.get("path", ""))
        name = Path(path).name.lower() if path else ""
        for pattern in self.deny_patterns:
            if name and fnmatch.fnmatch(name, pattern):
                return "deny", f"file name '{name}' matches the protected pattern '{pattern}'"
        if tool.risk == "read":
            return "allow", "read-only tool"
        if self.mode == "readonly":
            return "deny", "this session is read-only"
        if self.mode == "auto":
            return "allow", "writes inside the sandbox are allowed in auto mode"
        return "ask", "writes need approval in default mode"


def ask_human(tool_name: str, tool_input: dict, offline_approve: bool) -> bool:
    """Ask on the console. Without a console (or offline) the answer is no unless approved up front."""
    if offline_approve:
        return True
    if not sys.stdin or not sys.stdin.isatty():
        return False
    reply = input(f"\nAllow {tool_name} with {tool_input}? [y/N] ").strip().lower()
    return reply in ("y", "yes")


def add_write_tool(registry: lab_tools.ToolRegistry, workdir: Path) -> None:
    root = workdir.resolve()

    def write_text_file(args: dict) -> str:
        # Writes land in workspace/out/ so the sample files stay untouched.
        target = lab_tools.resolve_in_sandbox(root / "out", str(args["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        content = str(args["content"])
        if len(content) > 50_000:
            raise ValueError("content too large (max 50,000 characters)")
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to out/{args['path']}"

    registry.register(
        "write_text_file",
        "Write a UTF-8 text file into the workspace 'out' folder (overwrites).",
        {
            "path": {"type": "string", "description": "File name relative to the out folder"},
            "content": {"type": "string", "description": "Text to write"},
        },
        ["path", "content"],
        write_text_file,
        risk="write",
    )


def agent_loop(client, model_id, messages, registry, policy, approve_all=False, max_turns=8, verbose=True):
    usage: dict = {}
    for turn in range(1, max_turns + 1):
        response = client.converse(
            modelId=model_id,
            messages=messages,
            system=SYSTEM,
            inferenceConfig={"maxTokens": 1024},
            toolConfig=registry.tool_config(),
        )
        lc.add_usage(usage, response.get("usage"))
        message = response["output"]["message"]
        messages.append(message)
        if response["stopReason"] != "tool_use":
            return lc.message_text(message), response["stopReason"], usage

        results = []
        for call in lc.message_tool_uses(message):
            tool = registry.get(call["name"])
            if tool is None:
                output, is_error = registry.call(call["name"], call["input"])  # produces the "unknown tool" error
            else:
                decision, reason = policy.check(tool, call["input"])
                if decision == "ask" and approve_all:
                    decision, reason = "allow", "approved automatically (--approve or offline demo)"
                elif decision == "ask" and ask_human(call["name"], call["input"], False):
                    decision, reason = "allow", "approved by a human"
                elif decision == "ask":
                    decision, reason = "deny", "a human did not approve it"
                if decision == "deny":
                    output, is_error = f"PERMISSION DENIED: {reason}", True
                else:
                    output, is_error = registry.call(call["name"], call["input"])
                if verbose:
                    print(f"[turn {turn}] {call['name']} -> {decision.upper()} ({reason})")
            results.append(lc.tool_result_block(call["toolUseId"], output, is_error))
        messages.append({"role": "user", "content": results})
    return "(stopped: max_turns reached)", "max_turns", usage


def offline_script() -> list:
    return [
        lc.tool_use_response(("read_text_file", {"path": ".env"}), text="I will check the config first."),
        lc.tool_use_response(("write_text_file", {"path": "summary.txt", "content": "Orders upgrade is in progress"})),
        lc.text_response("I could not read .env (protected), but I wrote summary.txt as asked."),
    ]


def main(argv: list[str] | None = None) -> int:
    lc.load_env()
    parser = lc.common_parser(__doc__.splitlines()[0], DEFAULT_PROMPT)
    parser.add_argument("--mode", default="default", choices=["readonly", "default", "auto"])
    parser.add_argument("--approve", action="store_true", help="approve every 'ask' automatically (offline demos)")
    args = parser.parse_args(argv)

    workdir = lab_tools.default_workdir()
    registry = lab_tools.build_default_registry(workdir)
    add_write_tool(registry, workdir)
    policy = PermissionPolicy(args.mode)
    client = lc.make_runtime_client(args.offline, offline_script() if args.offline else None)
    model_id = lc.get_model_id(args.offline)
    print(f"Mode: {policy.mode} | tools: {', '.join(registry.names())}")
    try:
        text, stop, usage = agent_loop(
            client, model_id, [lc.user_text(args.prompt)], registry, policy,
            approve_all=args.approve or (args.offline and args.mode == "default"), max_turns=args.max_turns,
        )
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", lc.explain_aws_error(exc))
        return 1
    print("\nAnswer:", text)
    print("Stop:  ", stop, "| tokens:", usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
