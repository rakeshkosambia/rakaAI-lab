# b02: Tool use

> *Adding a tool means adding one handler.*

Equivalent in the Claude Code course: **s02 Tool Use**. Read `lab_tools.py` (repo root) next to this chapter: that file is the lesson; `code.py` is the demo.

## The problem

In b01 the tool was hard-wired into the loop. Every new tool would mean editing the loop, and one bad argument would crash it.

## The idea

A vending machine has a fixed panel of buttons (the **toolSpec**), and each button is wired to one mechanism (the **handler**). Add a button and a mechanism; the coin slot and the cash box (the loop) stay as they are.

`ToolRegistry` does four jobs so handlers can stay tiny:

1. `specs()` builds the `toolConfig` the model sees.
2. `call()` rejects unknown tools and missing arguments with a readable message.
3. It runs the handler and catches every exception.
4. It returns `(output, is_error)`, which becomes a `toolResult` (with `status: "error"` on failure).

Failures are **information for the model**, not crashes: it reads "no such file" and tries another path.

## Add a tool

```python
registry.register(
    "word_count",
    "Count the words in a workspace text file.",
    {"path": {"type": "string", "description": "File path relative to the workspace"}},
    ["path"],
    lambda args: str(len(read(args["path"]).split())),
)
```

That is the whole change. The loop is untouched.

## The sandbox

`resolve_in_sandbox()` resolves a path and refuses anything outside the workspace folder (`../secrets.txt`, absolute paths, symlink tricks). The offline script tries a path escape on purpose so you can watch it fail safely.

## Run it

```
python b02_tool_use/code.py --offline
run.bat b02_tool_use --prompt "Read notes.txt and total the quarterly volumes."
```

## Try this

1. Add `word_count` above in `build_default_registry` and ask the model to use it.
2. Make a tool return 500,000 characters. What should the handler do instead? (Look at how `read_text_file` truncates.)
3. Register the same name twice and read the error.

## Pitfalls

- **Vague descriptions.** The description is the model's only manual. "Does stuff" gets misused; say what it does, what it returns, and its limits.
- **Unbounded output.** Huge results burn context and money. Truncate and say that you truncated.
- **Trusting arguments.** Treat tool input like user input from the internet: validate paths, sizes and types.
- **JSON vs text results.** `toolResult` content can carry `{"text": ...}` or `{"json": {...}}`. The JSON form must be an object, not a bare list or number; `tool_result_block()` handles that choice.
