"""b08: Context compaction.

Motto: context always fills up; have a way to make room.

Three tools, cheapest first:
  1. micro_compact   replace OLD, long tool outputs with a short placeholder. No model call.
  2. summarize_compact  ask the model to summarise the oldest part of the conversation.
  3. (your call) stop and start a new session with a hand-written summary.

The trap this chapter is built around: a toolUse in one message must be answered by a
toolResult in the very next message. If you cut the history in the middle of such a pair,
the next converse() call fails with a ValidationException. So we only ever cut at a
"real" user message (one that contains text, not just tool results). find_tail_start does that.

Token counts here are an ESTIMATE (characters / 4). The real count comes back in
response["usage"]; use that to calibrate the threshold for your model.

Run:  python b08_context_compact/code.py --offline
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab_common as lc  # noqa: E402
import lab_tools  # noqa: E402

DEFAULT_PROMPT = "unused (this chapter builds a long demo conversation and compacts it)"
PLACEHOLDER = "[earlier tool output omitted to save context: {chars} characters]"
SUMMARY_SYSTEM = [{"text": "You compress conversations. Keep facts, decisions, file names, numbers and open questions. Be brief."}]


def estimate_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=True)) // 4


def is_tool_result_only(message: dict) -> bool:
    return message["role"] == "user" and bool(lc.message_tool_results(message)) and not lc.message_text(message)


def is_real_user_message(message: dict) -> bool:
    """A user message that starts a turn: has text and no toolResult blocks."""
    return message["role"] == "user" and bool(lc.message_text(message)) and not lc.message_tool_results(message)


def micro_compact(messages: list[dict], keep_recent: int = 2, min_chars: int = 300) -> list[dict]:
    """Shrink long text inside old toolResult blocks. IDs and statuses are kept, so pairing stays valid."""
    result = copy.deepcopy(messages)
    result_indexes = [i for i, m in enumerate(result) if lc.message_tool_results(m)]
    protected = set(result_indexes[-keep_recent:]) if keep_recent > 0 else set()
    for index in result_indexes:
        if index in protected:
            continue
        for block in result[index]["content"]:
            tool_result = block.get("toolResult")
            if not tool_result:
                continue
            for part in tool_result["content"]:
                if "text" in part and len(part["text"]) >= min_chars:
                    part["text"] = PLACEHOLDER.format(chars=len(part["text"]))
    return result


def find_tail_start(messages: list[dict], keep_last: int) -> int | None:
    """Index where the kept tail should begin: a real user message, searched backwards from the
    wanted position (keeps a bit more), then forwards (keeps a bit less). None if impossible."""
    wanted = max(1, len(messages) - keep_last)
    for index in range(wanted, 0, -1):
        if is_real_user_message(messages[index]):
            return index
    for index in range(wanted + 1, len(messages)):
        if is_real_user_message(messages[index]):
            return index
    return None


def render_transcript(messages: list[dict], max_result_chars: int = 400) -> str:
    lines = []
    for message in messages:
        for block in message["content"]:
            if "text" in block:
                lines.append(f"{message['role'].upper()}: {block['text']}")
            elif "toolUse" in block:
                lines.append(f"ASSISTANT called {block['toolUse']['name']} with {json.dumps(block['toolUse']['input'])}")
            elif "toolResult" in block:
                text = " ".join(p.get("text", json.dumps(p.get("json", ""))) for p in block["toolResult"]["content"])
                lines.append(f"TOOL RESULT: {text[:max_result_chars]}")
    return "\n".join(lines)


def summarize_compact(client, model_id: str, messages: list[dict], keep_last: int = 6) -> list[dict] | None:
    """Replace the old part of the history with a model-written summary. None if there is no safe cut."""
    start = find_tail_start(messages, keep_last)
    if start is None or start < 1:
        return None
    head, tail = messages[:start], messages[start:]
    prompt = "Summarise this conversation so work can continue from the summary alone:\n\n" + render_transcript(head)
    response = client.converse(
        modelId=model_id,
        messages=[lc.user_text(prompt)],
        system=SUMMARY_SYSTEM,
        inferenceConfig={"maxTokens": 800},
    )
    summary = lc.message_text(response["output"]["message"]).strip() or "(summary unavailable)"
    # roles must alternate: user summary, assistant acknowledgement, then the tail (starts with a user message)
    return [
        lc.user_text("[Summary of the earlier conversation]\n" + summary),
        lc.assistant_text("Understood. I will continue from this summary."),
        *tail,
    ]


def compact_if_needed(client, model_id, messages, threshold_tokens: int, keep_last: int = 6) -> tuple[list[dict], str]:
    """Returns (messages, how) where how is 'none', 'micro', 'summary' or 'micro-only'."""
    if estimate_tokens(messages) <= threshold_tokens:
        return messages, "none"
    shrunk = micro_compact(messages)
    if estimate_tokens(shrunk) <= threshold_tokens:
        return shrunk, "micro"
    summarised = summarize_compact(client, model_id, shrunk, keep_last)
    if summarised is None:
        return shrunk, "micro-only"
    return summarised, "summary"


def agent_loop(client, model_id, messages, registry, threshold_tokens=6000, max_turns=10):
    """The b02 loop with one extra line: compact before each model call."""
    usage: dict = {}
    for turn in range(1, max_turns + 1):
        messages[:], how = compact_if_needed(client, model_id, messages, threshold_tokens)
        if how != "none":
            print(f"[turn {turn}] compacted history ({how})")
        response = client.converse(
            modelId=model_id,
            messages=messages,
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
            output, is_error = registry.call(call["name"], call["input"])
            results.append(lc.tool_result_block(call["toolUseId"], output, is_error))
        messages.append({"role": "user", "content": results})
    return "(stopped: max_turns reached)", "max_turns", usage


def build_demo_conversation(rounds: int = 6, output_chars: int = 1500) -> list[dict]:
    """A realistic long history: each round is a user question, a tool call, a long tool result, an answer."""
    messages: list[dict] = []
    for n in range(1, rounds + 1):
        tool_use_id = f"demo_tooluse_{n}"
        messages.append(lc.user_text(f"Question {n}: please read report_{n}.txt and note the key figure."))
        messages.append(
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": tool_use_id, "name": "read_text_file", "input": {"path": f"report_{n}.txt"}}}]}
        )
        messages.append({"role": "user", "content": [lc.tool_result_block(tool_use_id, f"report {n} body " + "x" * output_chars)]})
        messages.append(lc.assistant_text(f"The key figure in report {n} is {n * 100}."))
    messages.append(lc.user_text("Now compare all of them briefly."))
    return messages


def is_valid_history(messages: list[dict], registry) -> tuple[bool, str]:
    """Check pairing and alternation with the same validator the offline fake uses."""
    try:
        lc.validate_converse_request({"modelId": "x", "messages": messages, "toolConfig": registry.tool_config()})
        return True, "valid"
    except lc.FakeValidationError as exc:
        return False, str(exc)


def main(argv: list[str] | None = None) -> int:
    lc.load_env()
    parser = lc.common_parser(__doc__.splitlines()[0], DEFAULT_PROMPT)
    parser.add_argument("--threshold", type=int, default=1000, help="estimated-token limit that triggers compaction")
    parser.add_argument("--break-it", action="store_true", help="show the classic bug: cut the history inside a tool pair")
    args = parser.parse_args(argv)

    registry = lab_tools.build_default_registry()
    history = build_demo_conversation()
    print(f"Demo history: {len(history)} messages, about {estimate_tokens(history)} tokens (estimated)")
    ok, why = is_valid_history(history, registry)
    print("Valid before compaction:", ok)

    if args.break_it:
        broken = history[2:]  # "keep the last N messages" cut in the middle of a tool pair
        print("Naive cut (history[2:]) valid?", is_valid_history(broken, registry))

    micro = micro_compact(history)
    print(f"\nAfter micro_compact: about {estimate_tokens(micro)} tokens, valid: {is_valid_history(micro, registry)[0]}")

    script = [lc.text_response("Reports 1-4 were read; key figures were 100, 200, 300 and 400.")]
    client = lc.make_runtime_client(args.offline, script if args.offline else None)
    model_id = lc.get_model_id(args.offline)
    try:
        compacted, how = compact_if_needed(client, model_id, history, args.threshold, keep_last=6)
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", lc.explain_aws_error(exc))
        return 1
    ok, why = is_valid_history(compacted, registry)
    print(f"After compact_if_needed ({how}): {len(compacted)} messages, about {estimate_tokens(compacted)} tokens, valid: {ok} ({why})")
    print("First message now starts with:", lc.message_text(compacted[0])[:60].replace("\n", " "), "...")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
