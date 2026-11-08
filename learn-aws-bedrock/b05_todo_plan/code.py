"""b05: Plan and todo.

Motto: an agent without a plan drifts.

Give the model one place to write its plan (the todo_write tool) and nudge it
when it forgets to update that plan. The plan lives in the HOST program, so you
can print it, save it, or show it to a human.

Rules enforced by the tool (the model learns them from the error messages):
  - statuses are pending / in_progress / completed
  - at most ONE item is in_progress at a time
  - at most 20 items

Nag: after NAG_AFTER model turns that use tools but do not touch the todo list,
a <reminder> is added to the tool results.

Run:  python b05_todo_plan/code.py --offline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab_common as lc  # noqa: E402
import lab_tools  # noqa: E402

DEFAULT_PROMPT = "Plan and do this: list the workspace, read notes.txt, total the quarterly volumes. Track your plan."
SYSTEM = [
    {
        "text": (
            "You are a careful assistant. For any task with 3 or more steps, first call todo_write with your plan, "
            "keep exactly one item in_progress, and mark items completed as you finish them. Keep answers short."
        )
    }
]
STATUSES = ("pending", "in_progress", "completed")
NAG_AFTER = 3
REMINDER = "<reminder>You have open todo items. Update the list with todo_write: mark finished items completed and set the next one in_progress.</reminder>"


class TodoList:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def update(self, items: list) -> str:
        if not isinstance(items, list) or not items:
            raise ValueError("items must be a non-empty list")
        if len(items) > 20:
            raise ValueError("too many items (max 20)")
        cleaned = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"item {index} must be an object with content and status")
            content = str(item.get("content", "")).strip()
            status = item.get("status")
            if not content:
                raise ValueError(f"item {index} has empty content")
            if status not in STATUSES:
                raise ValueError(f"item {index} has invalid status {status!r}; use one of {', '.join(STATUSES)}")
            cleaned.append({"content": content, "status": status})
        in_progress = sum(1 for item in cleaned if item["status"] == "in_progress")
        if in_progress > 1:
            raise ValueError(f"only one item may be in_progress, found {in_progress}")
        self.items = cleaned
        return self.render()

    def render(self) -> str:
        marks = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        return "\n".join(f"{marks[i['status']]} {i['content']}" for i in self.items) or "(no todos)"

    def has_open(self) -> bool:
        return any(item["status"] != "completed" for item in self.items)


def add_todo_tool(registry: lab_tools.ToolRegistry, todos: TodoList) -> None:
    def todo_write(args: dict) -> str:
        return todos.update(args["items"])

    registry.register(
        "todo_write",
        "Replace the whole todo list. Send every item each time.",
        {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": list(STATUSES)},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        ["items"],
        todo_write,
    )


def agent_loop(client, model_id, messages, registry, todos: TodoList, max_turns=10, verbose=True):
    usage: dict = {}
    turns_without_todo = 0
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
        used_todo = False
        for call in lc.message_tool_uses(message):
            output, is_error = registry.call(call["name"], call["input"])
            used_todo = used_todo or call["name"] == "todo_write"
            results.append(lc.tool_result_block(call["toolUseId"], output, is_error))
            if verbose:
                print(f"[turn {turn}] {call['name']}{' (ERROR)' if is_error else ''}")
        turns_without_todo = 0 if used_todo else turns_without_todo + 1
        if turns_without_todo >= NAG_AFTER and todos.has_open():
            results.append({"text": REMINDER})
            turns_without_todo = 0
            if verbose:
                print(f"[turn {turn}] nag: reminder added to the tool results")
        if used_todo and verbose:
            print(todos.render())
        messages.append({"role": "user", "content": results})
    return "(stopped: max_turns reached)", "max_turns", usage


def offline_script() -> list:
    plan = [
        {"content": "List the workspace", "status": "in_progress"},
        {"content": "Read notes.txt", "status": "pending"},
        {"content": "Total the quarterly volumes", "status": "pending"},
    ]
    done = [{"content": i["content"], "status": "completed"} for i in plan]
    return [
        lc.tool_use_response(("todo_write", {"items": plan})),
        lc.tool_use_response(("list_dir", {"path": "."})),
        lc.tool_use_response(("read_text_file", {"path": "notes.txt"})),
        lc.tool_use_response(("calculate", {"expression": "1250 + 1400 + 1310 + 1620"})),
        lc.tool_use_response(("todo_write", {"items": done})),
        lc.text_response("Done. The quarterly volumes total 5580 units."),
    ]


def main(argv: list[str] | None = None) -> int:
    lc.load_env()
    args = lc.common_parser(__doc__.splitlines()[0], DEFAULT_PROMPT).parse_args(argv)
    registry = lab_tools.build_default_registry()
    todos = TodoList()
    add_todo_tool(registry, todos)
    client = lc.make_runtime_client(args.offline, offline_script() if args.offline else None)
    model_id = lc.get_model_id(args.offline)
    try:
        text, stop, usage = agent_loop(client, model_id, [lc.user_text(args.prompt)], registry, todos, max(args.max_turns, 10))
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", lc.explain_aws_error(exc))
        return 1
    print("\nAnswer:", text)
    print("Final plan:\n" + todos.render())
    print("Stop:", stop, "| tokens:", usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
