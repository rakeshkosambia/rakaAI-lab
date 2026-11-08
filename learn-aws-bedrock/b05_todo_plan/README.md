# b05: Plan and todo

> *An agent without a plan drifts.*

Equivalent in the Claude Code course: **s05 TodoWrite**. Nothing here is AWS-specific; it matters on Bedrock because drift costs tokens.

## The problem

On a task with five or six steps, a model can lose the thread: it repeats a step, skips one, or wanders into side quests while the token meter runs.

## The idea

A pilot's checklist. Not because the pilot cannot fly, but because a written list that is **visible and updated** keeps long tasks honest. The list lives in your program, not only in the model's head, so you can print it, save it, or show it to a person.

The model uses one tool, `todo_write`, and sends the whole list each time:

```
[x] List the workspace
[>] Read notes.txt
[ ] Total the quarterly volumes
```

## The rules, enforced by the tool

- statuses: `pending`, `in_progress`, `completed`
- **at most one** item `in_progress`
- at most 20 items, no empty items

Break a rule and the tool returns a precise error, which the model reads and fixes. You do not need to put these rules in a long prompt.

## The nag

If the model makes `NAG_AFTER` (3) tool-using turns without touching the list while items are still open, the loop adds a short `<reminder>` text block next to the tool results. It is a nudge, not a command. Run offline and watch `nag: reminder added` appear on turn 4.

## Run it

```
python b05_todo_plan/code.py --offline
run.bat b05_todo_plan --prompt "Plan and do: list the workspace, read notes.txt, total the volumes."
```

## Try this

1. Set `NAG_AFTER = 1` and see how noisy that is.
2. Make the tool reject a plan with more than 8 items. Is a short plan better or worse for your tasks?
3. After the run, save `todos.render()` to a file. That is a cheap progress report.

## Pitfalls

- **Nag fatigue.** Too frequent reminders waste tokens and annoy the model into compliance without thought.
- **Stale plans.** If the task changes mid-way, the model must rewrite the plan; tell it so in the system prompt.
- **Plans for tiny tasks.** A one-step request does not need a todo list. The system prompt says to plan only for 3+ steps.
