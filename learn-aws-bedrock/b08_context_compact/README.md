# b08: Context compaction

> *Context always fills up; have a way to make room.*

Equivalent in the Claude Code course: **s08 Context Compact**. This is also where most real agent loops break, so the chapter is built around one bug.

## The problem

Every `converse()` call re-sends the whole history. Long tool outputs pile up, each turn costs more than the last, and eventually the request is too big for the model.

## The idea

Packing a suitcase for a long trip: first take out what is bulky and no longer needed, then roll up older clothes, and only if still too full, leave something behind.

1. **`micro_compact`**: replace *old* long tool outputs with a one-line placeholder. No model call. IDs and statuses stay, so tool pairs stay valid.
2. **`summarize_compact`**: ask the model to summarise the oldest part, keep the recent tail verbatim.
3. **Start fresh**: if neither works, end the session and begin a new one from a summary you wrote.

`compact_if_needed()` runs 1, then 2, and tells you which one it used: `none`, `micro`, `summary` or `micro-only`.

## The bug this chapter exists for

A `toolUse` in one message **must** be answered by a `toolResult` in the very next message. Cut the history at an arbitrary point (for example "keep the last 10 messages") and you can leave a `toolResult` whose `toolUse` was thrown away. Bedrock then rejects the request with a `ValidationException`, usually in the middle of a long run.

Reproduce it safely:

```
python b08_context_compact/code.py --offline --break-it
```

The fix is `find_tail_start()`: it only cuts at a **real user message** (one that has text and is not just tool results). After a cut, roles must still alternate, so the summary is a user message, followed by a short assistant acknowledgement, followed by the kept tail:

```
[user: summary] [assistant: "Understood..."] [user: real message] ... tail ...
```

## Run it

```
python b08_context_compact/code.py --offline
python b08_context_compact/code.py --offline --threshold 400
run.bat b08_context_compact
```

`agent_loop()` in `code.py` shows the one extra line that wires compaction into a real loop.

## Try this

1. Lower `--threshold` until `micro` is no longer enough and the `summary` path appears.
2. Lower `keep_last` to 2 and see how `find_tail_start` still lands on a valid cut.
3. In a live run, compare the estimate from `estimate_tokens()` with the real `usage` numbers, then adjust the threshold.

## Pitfalls

- **The estimate is rough** (characters divided by 4). The real counts are in `response["usage"]`; calibrate with them for your model.
- **A summary can lose facts.** Ask for names, numbers, file paths and open questions explicitly, as the prompt here does.
- **Summaries cost a model call.** Compact early and rarely, not on every turn.
- **Compaction hides evidence.** If you need an audit trail, keep the full history on disk (b04's log) before compacting.

## Verify on your own account

- Model context limits and whether your model supports **prompt caching** (a separate way to cut cost on repeated prefixes). Check the current Bedrock documentation for your model; this chapter does not depend on it.
