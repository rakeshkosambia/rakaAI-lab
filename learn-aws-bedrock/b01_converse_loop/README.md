# b01: The Converse loop

> *One loop. The model decides when to stop.*

Equivalent in the Claude Code course: **s01 Agent Loop**. Same idea, different door: instead of `client.messages.create`, you call Bedrock's `converse()` through boto3.

## The problem

A model answers once and stops. To "do things" it must be able to ask for an action, see the result, and keep going. Something has to run that cycle.

## The idea

Think of a cook and a runner. The cook (the model) is in a closed kitchen and can only read notes. The runner (your program) fetches whatever the cook asks for and brings back the result. The cook decides when the dish is done; the runner never decides that.

```
you --> messages[] --> converse() --> response
                                         |
                        stopReason == "tool_use"?
                          yes: run the tools, append results, call again
                          no:  return the text
```

## The Bedrock shapes you need

Request (the fields this course uses):

```python
client.converse(
    modelId=model_id,
    messages=messages,                       # [{"role": "user"|"assistant", "content": [blocks]}]
    system=[{"text": "..."}],                # optional instructions
    inferenceConfig={"maxTokens": 1024},
    toolConfig={"tools": [ {"toolSpec": {"name", "description", "inputSchema": {"json": {...JSON Schema...}}}} ]},
)
```

Response:

```python
response["output"]["message"]   # the assistant message; content has text blocks and/or toolUse blocks
response["stopReason"]          # "tool_use" means: run the tools and call again
response["usage"]               # inputTokens, outputTokens, totalTokens
```

A tool request inside the assistant message: `{"toolUse": {"toolUseId", "name", "input"}}`.
Your answer goes back as a **user** message: `{"toolResult": {"toolUseId", "content": [...], "status": "error"}}` (`status` only when it failed).

## The loop (from `code.py`)

```python
for turn in range(1, max_turns + 1):
    response = client.converse(...)
    message = response["output"]["message"]
    messages.append(message)                 # keep the assistant turn
    if response["stopReason"] != "tool_use":
        return lc.message_text(message)
    results = [run each toolUse and build a toolResult]
    messages.append({"role": "user", "content": results})
```

## Run it

```
python b01_converse_loop/code.py --offline
run.bat b01_converse_loop --prompt "What is 17 * 23 + 4?"
```

Offline mode replays a scripted conversation through a fake that rejects malformed requests exactly where Bedrock would, so the lesson is safe to read and run for free.

## Try this

1. Run with `--max-turns 1`. What does the loop return, and why?
2. Remove the line `messages.append(message)` and run offline. Read the error: it is the same mistake that bites real loops.
3. Ask for `10 / 0`. The tool fails; watch how the model recovers from an error result.

## Pitfalls

- **Forgetting to append the assistant message** before the tool results. Bedrock needs to see its own `toolUse` before your `toolResult`.
- **`stopReason == "max_tokens"`** means the answer was cut off; it is not an error. A cut-off `toolUse` block must not be executed.
- **Infinite loops.** `max_turns` is a hard stop. Keep it.
- **Cost.** Every turn re-sends the whole history. Long loops get expensive; b08 deals with that.

## Verify

- The full list of `stopReason` values and any model-specific limits: see the current Converse API reference.
- Which models support tool use through Converse: check the Bedrock "supported models and features" page for your model.
- No `temperature` is set on purpose: supported inference parameters vary by model. Add it only after checking your model.
