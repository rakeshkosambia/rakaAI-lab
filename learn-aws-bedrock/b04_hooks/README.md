# b04: Hooks

> *Hook around the loop; never rewrite the loop.*

Equivalent in the Claude Code course: **s04 Hooks**. AWS twist: guardrails and cost visibility matter more when every call is metered.

## The problem

You need logging, a spending cap, secret masking and content checks. If each goes inside the loop body, the loop becomes a tangle and every chapter after this one inherits the mess.

## The idea

Airport security: the flight (the loop) is unchanged, but there are checkpoints before boarding and after landing. Each checkpoint can stop a passenger or inspect luggage without knowing how planes fly.

Four hook points:

| Hook | Runs | Can do |
|---|---|---|
| `before_model(ctx)` | before every `converse()` | raise to stop the run (budget, guardrail) |
| `after_model(ctx, response)` | after every reply | record usage and stop reason |
| `before_tool(ctx, name, input)` | before a tool | return a reason string to **block** the call |
| `after_tool(ctx, name, output, is_error)` | after a tool | return a **rewritten** output (redaction) |

## The sample hooks

- **AuditLog**: appends one JSON line per event to `logs/audit.jsonl`. Tool inputs are redacted and truncated; tool *outputs* are logged only by size.
- **TokenBudget**: stops the run when `usage` reaches `LAB_MAX_TOTAL_TOKENS`. If you set `LAB_PRICE_IN_PER_1K` / `LAB_PRICE_OUT_PER_1K`, it prints a cost estimate. No prices are hard-coded because they change.
- **RedactSecrets**: masks AWS key IDs, `sk-ant-` style keys and `secret=...` / `password: ...` patterns in tool output **before the model sees it**. The demo tool `show_config` returns a fake secret; the run prints whether it leaked (it should say `False`).
- **GuardrailCheck**: if `BEDROCK_GUARDRAIL_ID` is set, sends the newest user text to Bedrock's `apply_guardrail` (`source="INPUT"`) and stops the run when the guardrail intervenes.

Installing them takes three lines and the loop does not change:

```python
hooks = HookRunner()
AuditLog(path).install(hooks)
TokenBudget(50_000).install(hooks)
```

## Run it

```
python b04_hooks/code.py --offline
run.bat b04_hooks
```

Then open `logs/audit.jsonl` (ignored by git) and read the trail.

## Try this

1. Set `LAB_MAX_TOTAL_TOKENS=100` and run offline. The run stops with `budget`.
2. Write a `before_tool` hook that blocks `calculate` when the expression contains `**`.
3. Add a new secret pattern (for example a JWT) to `SECRET_PATTERNS` and add a test for it in `tests/`.

## Pitfalls

- **Redaction is a safety net, not a vault.** Regexes miss things. Keep real secrets out of tool output in the first place.
- **Audit logs can leak.** Log sizes and names, not raw content, unless you have decided otherwise.
- **A hook that raises on every call** looks like a hung model. Make errors loud and specific.
- Hooks run in the order added: put `RedactSecrets` before anything that stores output.

## Verify on your own account

- The exact request/response shape of `apply_guardrail` and the guardrail version names: check the current Bedrock Guardrails API reference. This chapter's code follows my understanding of it and is only tested against a stub.
- Bedrock also has account-level **model invocation logging**; look it up in the Bedrock documentation if you need a provider-side audit trail in addition to this local one.
