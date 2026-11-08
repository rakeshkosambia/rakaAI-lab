"""b04: Hooks.

Motto: hook around the loop, never rewrite the loop.

Four hook points, so cross-cutting concerns live outside the loop body:
  before_model(ctx)                        may raise to stop the run (budget, guardrail)
  after_model(ctx, response)               observe usage and stop reason
  before_tool(ctx, name, input) -> reason  return a string to block the call
  after_tool(ctx, name, output, is_err) -> output   may rewrite the output (redaction)

Sample hooks: AuditLog (JSONL file), TokenBudget (token cap + cost estimate),
RedactSecrets (mask keys before the model sees them), GuardrailCheck (Bedrock
Guardrails apply_guardrail on the user's input).

Run:  python b04_hooks/code.py --offline
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab_common as lc  # noqa: E402
import lab_tools  # noqa: E402

DEFAULT_PROMPT = "Show me the config, then compute (1250 * 12) / 4."
SYSTEM = [{"text": "You are a careful assistant. Use tools instead of guessing. Keep answers short."}]


class BudgetExceeded(RuntimeError):
    pass


class GuardrailBlocked(RuntimeError):
    pass


@dataclass
class RunContext:
    client: Any
    model_id: str
    messages: list
    usage: dict = field(default_factory=dict)
    turn: int = 0


class HookRunner:
    EVENTS = ("before_model", "after_model", "before_tool", "after_tool")

    def __init__(self) -> None:
        self.hooks: dict[str, list[Callable]] = {event: [] for event in self.EVENTS}

    def add(self, event: str, fn: Callable) -> "HookRunner":
        if event not in self.hooks:
            raise ValueError(f"unknown hook event: {event}")
        self.hooks[event].append(fn)
        return self

    def before_model(self, ctx: RunContext) -> None:
        for fn in self.hooks["before_model"]:
            fn(ctx)

    def after_model(self, ctx: RunContext, response: dict) -> None:
        for fn in self.hooks["after_model"]:
            fn(ctx, response)

    def before_tool(self, ctx: RunContext, name: str, tool_input: dict) -> str | None:
        for fn in self.hooks["before_tool"]:
            reason = fn(ctx, name, tool_input)
            if reason:
                return reason
        return None

    def after_tool(self, ctx: RunContext, name: str, output: Any, is_error: bool) -> Any:
        for fn in self.hooks["after_tool"]:
            output = fn(ctx, name, output, is_error)
        return output


# --------------------------------------------------------------------------
# Sample hooks
# --------------------------------------------------------------------------
SECRET_PATTERNS = [
    (re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED-AWS-KEY-ID]"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "[REDACTED-API-KEY]"),
    (re.compile(r"(?i)((?:secret|password|passwd|token|api[_-]?key)[A-Za-z0-9_\-]*\s*[=:]\s*)\S+"), r"\1[REDACTED]"),
]


def redact(text: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RedactSecrets:
    """after_tool: mask anything that looks like a credential before the model sees it."""

    def after_tool(self, ctx, name, output, is_error):
        return redact(output) if isinstance(output, str) else output

    def install(self, hooks: HookRunner) -> None:
        hooks.add("after_tool", self.after_tool)


class AuditLog:
    """Append one JSON line per event. Inputs are redacted and truncated."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event: str, **fields: Any) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def before_model(self, ctx):
        self._write("model_call", turn=ctx.turn, messages=len(ctx.messages))

    def after_model(self, ctx, response):
        self._write("model_reply", turn=ctx.turn, stopReason=response.get("stopReason"), usage=response.get("usage"))

    def before_tool(self, ctx, name, tool_input):
        self._write("tool_call", turn=ctx.turn, tool=name, input=redact(json.dumps(tool_input))[:300])
        return None

    def after_tool(self, ctx, name, output, is_error):
        self._write("tool_result", turn=ctx.turn, tool=name, error=is_error, chars=len(str(output)))
        return output

    def install(self, hooks: HookRunner) -> None:
        hooks.add("before_model", self.before_model).add("after_model", self.after_model)
        hooks.add("before_tool", self.before_tool).add("after_tool", self.after_tool)


class TokenBudget:
    """Stop the run once it has used max_total tokens. Optionally estimate cost."""

    def __init__(self, max_total: int, price_in_per_1k: float | None = None, price_out_per_1k: float | None = None):
        self.max_total = max_total
        self.price_in = price_in_per_1k
        self.price_out = price_out_per_1k

    def before_model(self, ctx):
        used = lc.total_tokens(ctx.usage)
        if used >= self.max_total:
            raise BudgetExceeded(f"token budget reached: {used} >= {self.max_total}")

    def estimate_cost(self, usage: dict) -> float | None:
        if self.price_in is None or self.price_out is None:
            return None
        return usage.get("inputTokens", 0) / 1000 * self.price_in + usage.get("outputTokens", 0) / 1000 * self.price_out

    def install(self, hooks: HookRunner) -> None:
        hooks.add("before_model", self.before_model)


class GuardrailCheck:
    """before_model: run the newest user text through Bedrock Guardrails (apply_guardrail)."""

    def __init__(self, guardrail_id: str, version: str = "DRAFT"):
        self.guardrail_id = guardrail_id
        self.version = version

    def before_model(self, ctx):
        last = ctx.messages[-1]
        text = lc.message_text(last) if last["role"] == "user" else ""
        if not text:
            return
        response = ctx.client.apply_guardrail(
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.version,
            source="INPUT",
            content=[{"text": {"text": text}}],
        )
        if response.get("action") == "GUARDRAIL_INTERVENED":
            raise GuardrailBlocked("the guardrail blocked this input")

    def install(self, hooks: HookRunner) -> None:
        hooks.add("before_model", self.before_model)


# --------------------------------------------------------------------------
# The loop: identical to b02 except for the hook calls
# --------------------------------------------------------------------------
def agent_loop(client, model_id, messages, registry, hooks: HookRunner, max_turns=8):
    ctx = RunContext(client=client, model_id=model_id, messages=messages)
    try:
        for turn in range(1, max_turns + 1):
            ctx.turn = turn
            hooks.before_model(ctx)
            response = client.converse(
                modelId=model_id,
                messages=messages,
                system=SYSTEM,
                inferenceConfig={"maxTokens": 1024},
                toolConfig=registry.tool_config(),
            )
            lc.add_usage(ctx.usage, response.get("usage"))
            hooks.after_model(ctx, response)
            message = response["output"]["message"]
            messages.append(message)
            if response["stopReason"] != "tool_use":
                return lc.message_text(message), response["stopReason"], ctx.usage

            results = []
            for call in lc.message_tool_uses(message):
                blocked = hooks.before_tool(ctx, call["name"], call["input"])
                if blocked:
                    output, is_error = f"BLOCKED BY HOOK: {blocked}", True
                else:
                    output, is_error = registry.call(call["name"], call["input"])
                    output = hooks.after_tool(ctx, call["name"], output, is_error)
                results.append(lc.tool_result_block(call["toolUseId"], output, is_error))
            messages.append({"role": "user", "content": results})
        return "(stopped: max_turns reached)", "max_turns", ctx.usage
    except BudgetExceeded as exc:
        return f"(stopped: {exc})", "budget", ctx.usage
    except GuardrailBlocked as exc:
        return f"(stopped: {exc})", "guardrail", ctx.usage


def add_demo_config_tool(registry: lab_tools.ToolRegistry) -> None:
    """A demo tool whose output contains a fake secret, to show redaction working."""

    def show_config(args: dict) -> str:
        return "region=ap-southeast-1\nmodel_alias=demo\naws_secret_access_key = EXAMPLE-NOT-A-REAL-SECRET\n"

    registry.register("show_config", "Show the demo application configuration.", {}, [], show_config)


def build_hooks(log_path: Path, max_total: int) -> tuple[HookRunner, TokenBudget]:
    hooks = HookRunner()
    budget = TokenBudget(
        max_total,
        float(os.getenv("LAB_PRICE_IN_PER_1K") or "0") or None,
        float(os.getenv("LAB_PRICE_OUT_PER_1K") or "0") or None,
    )
    AuditLog(log_path).install(hooks)
    budget.install(hooks)
    RedactSecrets().install(hooks)
    return hooks, budget


def offline_script() -> list:
    return [
        lc.tool_use_response(("show_config", {})),
        lc.tool_use_response(("calculate", {"expression": "(1250 * 12) / 4"})),
        lc.text_response("Config shown (the secret was masked). The calculation gives 3750.0."),
    ]


def main(argv: list[str] | None = None) -> int:
    lc.load_env()
    args = lc.common_parser(__doc__.splitlines()[0], DEFAULT_PROMPT).parse_args(argv)
    registry = lab_tools.build_default_registry()
    add_demo_config_tool(registry)
    log_path = lc.ROOT / "logs" / "audit.jsonl"
    hooks, budget = build_hooks(log_path, int(os.getenv("LAB_MAX_TOTAL_TOKENS") or "50000"))
    client = lc.make_runtime_client(args.offline, offline_script() if args.offline else None)
    guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID", "").strip()
    if guardrail_id and not args.offline:
        GuardrailCheck(guardrail_id, os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")).install(hooks)
        print("Guardrail check enabled.")
    model_id = lc.get_model_id(args.offline)
    messages = [lc.user_text(args.prompt)]
    try:
        text, stop, usage = agent_loop(client, model_id, messages, registry, hooks, args.max_turns)
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", lc.explain_aws_error(exc))
        return 1
    print("\nAnswer:", text)
    print("Stop:  ", stop, "| tokens:", usage)
    cost = budget.estimate_cost(usage)
    if cost is not None:
        print(f"Estimated cost: {cost:.6f} (using your LAB_PRICE_* values)")
    print("Secret leaked into the conversation:", "EXAMPLE-NOT-A-REAL-SECRET" in json.dumps(messages))
    print("Audit log:", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
