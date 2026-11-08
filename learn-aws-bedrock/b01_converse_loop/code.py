"""b01: The Converse loop.

Motto: one loop; the model decides when to stop.

Send messages to the model. If it asks for a tool (stopReason == "tool_use"),
run the tool, send the result back, and go around again. Otherwise return.
The loop never changes in later chapters; mechanisms attach around it.

Run:  python b01_converse_loop/code.py --offline
      python b01_converse_loop/code.py --prompt "What is 17 * 23 + 4?"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab_common as lc  # noqa: E402

DEFAULT_PROMPT = "What is 17 * 23, plus 4? Use the calculator."
SYSTEM = [{"text": "You are a careful assistant. Use the calculate tool for any arithmetic. Keep answers short."}]
TOOLS = [
    lc.tool_spec(
        "calculate",
        "Evaluate an arithmetic expression such as '17 * 23'.",
        {"expression": {"type": "string", "description": "The arithmetic expression"}},
        ["expression"],
    )
]


def run_tool(name: str, tool_input: dict) -> str:
    if name == "calculate":
        return lc.safe_calculate(tool_input["expression"])
    raise KeyError(f"unknown tool: {name}")


def agent_loop(client, model_id: str, messages: list[dict], max_turns: int = 8, verbose: bool = True):
    """Run the loop. Returns (final_text, stop_reason, usage_totals)."""
    usage: dict = {}
    for turn in range(1, max_turns + 1):
        response = client.converse(
            modelId=model_id,
            messages=messages,
            system=SYSTEM,
            inferenceConfig={"maxTokens": 1024},
            toolConfig={"tools": TOOLS},
        )
        lc.add_usage(usage, response.get("usage"))
        message = response["output"]["message"]
        messages.append(message)  # always keep the assistant turn in history
        stop_reason = response["stopReason"]
        if verbose:
            print(f"[turn {turn}] stopReason={stop_reason} usage={response.get('usage')}")

        if stop_reason != "tool_use":
            # end_turn is the normal finish. max_tokens means the answer was cut off.
            return lc.message_text(message), stop_reason, usage

        results = []
        for call in lc.message_tool_uses(message):
            try:
                output = run_tool(call["name"], call["input"])
                results.append(lc.tool_result_block(call["toolUseId"], output))
                if verbose:
                    print(f"         tool {call['name']}({call['input']}) -> {output}")
            except Exception as exc:  # noqa: BLE001 - let the model see the failure
                results.append(lc.tool_result_block(call["toolUseId"], f"{type(exc).__name__}: {exc}", is_error=True))
                if verbose:
                    print(f"         tool {call['name']} FAILED: {exc}")
        messages.append({"role": "user", "content": results})

    return "(stopped: max_turns reached)", "max_turns", usage


def offline_script() -> list:
    return [
        lc.tool_use_response(("calculate", {"expression": "17 * 23 + 4"}), text="Let me calculate that."),
        lc.text_response("17 * 23 + 4 = 395."),
    ]


def main(argv: list[str] | None = None) -> int:
    lc.load_env()
    args = lc.common_parser(__doc__.splitlines()[0], DEFAULT_PROMPT).parse_args(argv)
    client = lc.make_runtime_client(args.offline, offline_script() if args.offline else None)
    model_id = lc.get_model_id(args.offline)
    try:
        text, stop, usage = agent_loop(client, model_id, [lc.user_text(args.prompt)], args.max_turns)
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", lc.explain_aws_error(exc))
        return 1
    print("\nAnswer:", text)
    print("Stop:  ", stop, "| tokens:", usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
