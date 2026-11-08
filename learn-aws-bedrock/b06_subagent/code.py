"""b06: Subagents.

Motto: big tasks split small, and each subtask gets a clean context.

A subagent is the same loop run again with a fresh messages list. The parent
sees ONLY the subagent's final text, not its tool calls or file contents, so the
parent's context stays small. The subagent gets a restricted tool set and no
`task` tool, so it cannot spawn more subagents (depth is capped at 1).

Token usage from the subagent rolls into one shared total, so budgets still work.

Run:  python b06_subagent/code.py --offline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab_common as lc  # noqa: E402
import lab_tools  # noqa: E402

DEFAULT_PROMPT = "Use a subagent to read notes.txt and report the quarterly volumes and their total. Then tell me the busiest quarter."
PARENT_SYSTEM = [
    {
        "text": (
            "You are a coordinator. For anything that needs reading files, delegate to the task tool with a clear, "
            "self-contained prompt, then use the subagent's report. Keep answers short."
        )
    }
]
CHILD_SYSTEM = [{"text": "You are a focused worker. Do exactly the task given, use tools as needed, then reply with a short factual report."}]
CHILD_TOOLS = ["calculate", "list_dir", "read_text_file"]
MAX_REPORT_CHARS = 4000


def run_loop(client, model_id, messages, registry, system, usage: dict, max_turns: int, label: str, verbose: bool = True):
    """The same loop as b02. Adds into the shared usage dict. Returns (text, stop_reason)."""
    for turn in range(1, max_turns + 1):
        response = client.converse(
            modelId=model_id,
            messages=messages,
            system=system,
            inferenceConfig={"maxTokens": 1024},
            toolConfig=registry.tool_config(),
        )
        lc.add_usage(usage, response.get("usage"))
        message = response["output"]["message"]
        messages.append(message)
        if response["stopReason"] != "tool_use":
            return lc.message_text(message), response["stopReason"]
        results = []
        for call in lc.message_tool_uses(message):
            output, is_error = registry.call(call["name"], call["input"])
            results.append(lc.tool_result_block(call["toolUseId"], output, is_error))
            if verbose:
                print(f"  [{label} turn {turn}] {call['name']}{' (ERROR)' if is_error else ''}")
        messages.append({"role": "user", "content": results})
    return "(subagent stopped: max_turns reached)", "max_turns"


def add_task_tool(parent_registry, client, model_id, usage: dict, verbose: bool = True) -> None:
    """Register `task`. The child registry is built from CHILD_TOOLS, so it never contains `task`."""
    child_registry = parent_registry.subset(CHILD_TOOLS)

    def task(args: dict) -> str:
        prompt = str(args["prompt"]).strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        if verbose:
            print(f"  [main] spawning subagent: {prompt[:80]}")
        child_messages = [lc.user_text(prompt)]  # fresh context: nothing from the parent
        text, stop = run_loop(client, model_id, child_messages, child_registry, CHILD_SYSTEM, usage, 6, "sub", verbose)
        if verbose:
            print(f"  [main] subagent finished in {len(child_messages)} messages (parent will see only its final text)")
        report = text or "(subagent produced no text)"
        if stop != "end_turn":
            report += f"\n[note: subagent ended with stopReason={stop}]"
        return report[:MAX_REPORT_CHARS]

    parent_registry.register(
        "task",
        "Delegate a self-contained subtask to a subagent with a clean context. Returns the subagent's final report.",
        {"prompt": {"type": "string", "description": "Complete instructions for the subagent"}},
        ["prompt"],
        task,
    )


def offline_script() -> list:
    return [
        lc.tool_use_response(
            ("task", {"prompt": "Read notes.txt in the workspace and report the four quarterly volumes and their total."})
        ),
        lc.tool_use_response(("read_text_file", {"path": "notes.txt"})),  # the subagent's first call
        lc.text_response("Q1 1250, Q2 1400, Q3 1310, Q4 1620. Total 5580."),  # the subagent's report
        lc.text_response("The busiest quarter is Q4 with 1620 units (total across the year: 5580)."),  # the parent's answer
    ]


def main(argv: list[str] | None = None) -> int:
    lc.load_env()
    args = lc.common_parser(__doc__.splitlines()[0], DEFAULT_PROMPT).parse_args(argv)
    client = lc.make_runtime_client(args.offline, offline_script() if args.offline else None)
    model_id = lc.get_model_id(args.offline)
    usage: dict = {}
    registry = lab_tools.build_default_registry()
    add_task_tool(registry, client, model_id, usage)
    messages = [lc.user_text(args.prompt)]
    try:
        text, stop = run_loop(client, model_id, messages, registry, PARENT_SYSTEM, usage, args.max_turns, "main")
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", lc.explain_aws_error(exc))
        return 1
    print("\nAnswer:", text)
    print(f"Parent context: {len(messages)} messages | stop: {stop} | total tokens incl. subagent: {usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
