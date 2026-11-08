# b06: Subagents

> *Big tasks split small, and each subtask gets a clean context.*

Equivalent in the Claude Code course: **s06 Subagent**. Later, chapter b14 shows the AWS-flavoured ways to do multi-agent work (Strands patterns, AgentCore).

## The problem

Reading ten files to answer one question stuffs the conversation with nine files you no longer need. Every later turn re-sends all of it, so you pay for it again and again, and the model gets distracted by it.

## The idea

A manager who sends an assistant to the archive. The assistant comes back with a one-page report; the manager never sees the boxes of paper. In code:

```python
child_messages = [lc.user_text(prompt)]          # fresh context: nothing from the parent
text, stop = run_loop(client, model_id, child_messages, child_registry, ...)
return text                                      # the parent receives ONLY this
```

A subagent is simply **the same loop run again** with a new `messages` list and a smaller tool set.

## Design choices in `code.py`

- **No recursion.** The child's registry is built from `CHILD_TOOLS` and does not contain `task`, so a subagent cannot spawn subagents (depth cap 1).
- **Read-only by default.** The child gets `calculate`, `list_dir`, `read_text_file`. Give a child write tools deliberately, not by accident.
- **Shared usage total.** The child adds into the same `usage` dict, so a token budget (b04) still sees everything.
- **Capped report.** The child's text is truncated to 4,000 characters so a runaway child cannot flood the parent.
- **Honest endings.** If the child ends abnormally (for example `max_turns`), the report says so.

Run offline and compare: the parent ends with 4 messages while the subagent used 4 of its own.

## Run it

```
python b06_subagent/code.py --offline
run.bat b06_subagent --prompt "Use a subagent to read notes.txt and list the open items."
```

## Try this

1. Print `len(child_messages)` and `len(parent_messages)` after a live run. How much context did you save?
2. Give the subagent a different system prompt (for example "reply in exactly 3 bullet points") and watch the parent's behaviour.
3. Run two subagents at once with threads. Mind Bedrock rate limits: parallel calls raise throttling errors sooner.

## Pitfalls

- **Vague delegation.** The child knows nothing but its prompt. "Check that thing we discussed" fails; write a complete, self-contained instruction.
- **Child results are untrusted text.** If a child reads a file that contains instructions, its report may carry them back. Treat reports as data.
- **Cost is not free.** A subagent saves *parent* context but still spends tokens. Use it for noisy work, not for everything.
