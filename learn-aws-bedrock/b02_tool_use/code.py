"""b02: Tool use.

Motto: adding a tool means adding one handler.

The loop from b01 is unchanged. What is new is lab_tools.ToolRegistry: it builds
the toolSpec the model sees, checks arguments, runs the handler, and turns every
failure into an error result. File tools are confined to one sandbox folder.

Read lab_tools.py alongside this file: that is where the lesson lives.

Run:  python b02_tool_use/code.py --offline
      python b02_tool_use/code.py --prompt "Read notes.txt and total the quarterly volumes."
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab_common as lc  # noqa: E402
import lab_tools  # noqa: E402

DEFAULT_PROMPT = "Look in the workspace, read notes.txt, and add up the four quarterly volumes."
SYSTEM = [{"text": "You are a careful assistant with file and calculator tools. Use tools instead of guessing. Keep answers short."}]


def agent_loop(client, model_id: str, messages: list[dict], registry: lab_tools.ToolRegistry, max_turns: int = 8, verbose: bool = True):
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
        stop_reason = response["stopReason"]
        if verbose:
            print(f"[turn {turn}] stopReason={stop_reason}")
        if stop_reason != "tool_use":
            return lc.message_text(message), stop_reason, usage

        results = []
        for call in lc.message_tool_uses(message):
            output, is_error = registry.call(call["name"], call["input"])
            results.append(lc.tool_result_block(call["toolUseId"], output, is_error))
            if verbose:
                shown = str(output).replace("\n", " | ")[:100]
                print(f"         {'ERROR ' if is_error else ''}{call['name']}({call['input']}) -> {shown}")
        messages.append({"role": "user", "content": results})
    return "(stopped: max_turns reached)", "max_turns", usage


def offline_script() -> list:
    return [
        lc.tool_use_response(("list_dir", {"path": "."})),
        lc.tool_use_response(("read_text_file", {"path": "notes.txt"})),
        lc.tool_use_response(("read_text_file", {"path": "../secrets.txt"}), text="Let me also try a file outside the workspace."),
        lc.tool_use_response(("calculate", {"expression": "1250 + 1400 + 1310 + 1620"})),
        lc.text_response("The four quarterly volumes add up to 5580 units. (The file outside the workspace was correctly refused.)"),
    ]


def main(argv: list[str] | None = None) -> int:
    lc.load_env()
    args = lc.common_parser(__doc__.splitlines()[0], DEFAULT_PROMPT).parse_args(argv)
    client = lc.make_runtime_client(args.offline, offline_script() if args.offline else None)
    model_id = lc.get_model_id(args.offline)
    registry = lab_tools.build_default_registry()
    print("Tools:", ", ".join(registry.names()))
    try:
        text, stop, usage = agent_loop(client, model_id, [lc.user_text(args.prompt)], registry, args.max_turns)
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", lc.explain_aws_error(exc))
        return 1
    print("\nAnswer:", text)
    print("Stop:  ", stop, "| tokens:", usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
