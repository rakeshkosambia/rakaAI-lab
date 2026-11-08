"""Shared helpers for the learn-aws-bedrock lessons.

Everything here is small on purpose. The chapters (b00 .. b08) import this file
for four things:

1. Building a boto3 ``bedrock-runtime`` client from .env settings (live mode).
2. A ``FakeBedrockRuntime`` that replays scripted responses (offline mode), and
   checks every request the way the real Converse API would reject a bad one.
3. Small message helpers for the Converse message shape.
4. A safe arithmetic evaluator used as the first demo tool.

Offline mode lets you read and run every lesson without an AWS account and
without spending tokens. Live mode uses your own AWS profile.
"""
from __future__ import annotations

import argparse
import ast
import copy
import operator
import os
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
OFFLINE_MODEL_ID = "offline-fake-model"


# --------------------------------------------------------------------------
# Settings and clients
# --------------------------------------------------------------------------
def load_env() -> None:
    """Load .env from the repo root if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def make_session():
    """Create a boto3 Session from AWS_PROFILE / AWS_REGION (both optional)."""
    import boto3

    profile = os.getenv("AWS_PROFILE") or None
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or None
    return boto3.Session(profile_name=profile, region_name=region)


def make_runtime_client(offline: bool, script: list | None = None):
    """Return a bedrock-runtime client (live) or a scripted fake (offline)."""
    if offline:
        return FakeBedrockRuntime(script or [])
    from botocore.config import Config

    config = Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=120)
    return make_session().client("bedrock-runtime", config=config)


def get_model_id(offline: bool) -> str:
    """Model ID or inference-profile ID from BEDROCK_MODEL_ID."""
    if offline:
        return OFFLINE_MODEL_ID
    model_id = os.getenv("BEDROCK_MODEL_ID", "").strip()
    if not model_id:
        raise SystemExit(
            "BEDROCK_MODEL_ID is not set.\n"
            "Run b00_setup/code.py to list the models your account can call, "
            "then put one ID in your .env file."
        )
    return model_id


def common_parser(description: str, default_prompt: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use scripted fake responses: no AWS account, no cost",
    )
    parser.add_argument("--prompt", default=default_prompt, help="what to ask the agent")
    parser.add_argument("--max-turns", type=int, default=8, help="safety cap on model calls")
    return parser


def explain_aws_error(exc: Exception) -> str:
    """Turn common AWS / botocore errors into a hint a human can act on."""
    name = type(exc).__name__
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
    text = str(exc)
    lowered = text.lower()

    if "expired" in lowered or code in ("ExpiredTokenException", "ExpiredToken") or name == "TokenRetrievalError":
        return "Your AWS credentials or SSO session expired. Log in again with your usual SSO login method, then retry."
    if name in ("NoCredentialsError", "ProfileNotFound") or code == "UnrecognizedClientException":
        return "AWS credentials were not found or are not valid. Check AWS_PROFILE in .env and that you are logged in."
    if name == "NoRegionError":
        return "No AWS region is set. Put AWS_REGION in .env."
    if code == "AccessDeniedException":
        return (
            "Access denied. Check (1) model access is enabled for this model in this Region, "
            "and (2) your IAM role allows bedrock:InvokeModel on it."
        )
    if code == "ValidationException":
        return (
            "Bedrock rejected the request. Common causes: wrong model ID, a model that needs an "
            "inference profile ID instead, a model not offered in this Region, or an unsupported "
            "setting. Run b00_setup/code.py to see valid IDs."
        )
    if code in ("ThrottlingException", "ServiceQuotaExceededException"):
        return "Rate or quota limit reached. Wait a moment and retry, or lower the request rate."
    if code in ("ResourceNotFoundException", "ModelNotReadyException"):
        return "The model was not found or is not ready in this Region. Check the model ID and Region."
    return f"{name}: {text}"


# --------------------------------------------------------------------------
# Converse message helpers
# --------------------------------------------------------------------------
def user_text(text: str) -> dict:
    return {"role": "user", "content": [{"text": text}]}


def assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


def message_text(message: dict) -> str:
    return "".join(block["text"] for block in message.get("content", []) if "text" in block)


def message_tool_uses(message: dict) -> list[dict]:
    return [block["toolUse"] for block in message.get("content", []) if "toolUse" in block]


def message_tool_results(message: dict) -> list[dict]:
    return [block["toolResult"] for block in message.get("content", []) if "toolResult" in block]


def tool_result_block(tool_use_id: str, payload: Any, is_error: bool = False) -> dict:
    """Build a toolResult content block. Strings go as text, dicts as json."""
    if isinstance(payload, dict):
        content = [{"json": payload}]
    else:
        content = [{"text": str(payload)}]
    block: dict[str, Any] = {"toolUseId": tool_use_id, "content": content}
    if is_error:
        block["status"] = "error"
    return {"toolResult": block}


def tool_spec(name: str, description: str, properties: dict | None = None, required: list | None = None) -> dict:
    """Build one entry for toolConfig['tools']."""
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = list(required)
    return {"toolSpec": {"name": name, "description": description, "inputSchema": {"json": schema}}}


def add_usage(total: dict, usage: dict | None) -> dict:
    """Add a response's usage block into a running total (in place)."""
    usage = usage or {}
    for key in ("inputTokens", "outputTokens", "totalTokens"):
        total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)
    return total


def total_tokens(usage: dict) -> int:
    if usage.get("totalTokens"):
        return int(usage["totalTokens"])
    return int(usage.get("inputTokens", 0)) + int(usage.get("outputTokens", 0))


# --------------------------------------------------------------------------
# Offline fake of bedrock-runtime
# --------------------------------------------------------------------------
class FakeValidationError(ValueError):
    """Raised by the fake where the real API would return a ValidationException."""


def validate_converse_request(request: dict) -> None:
    """Reject requests the real Converse API would reject.

    This is a teaching aid, not a full copy of the service rules. It checks the
    mistakes that bite agent loops most often:
    - roles must start with user and alternate
    - every toolUse must be answered by a toolResult with the same ID in the very next message
    - a toolResult must answer a toolUse from the message right before it
    - messages that contain tool blocks need toolConfig in the request
    """
    if not request.get("modelId"):
        raise FakeValidationError("modelId is required")
    messages = request.get("messages")
    if not messages:
        raise FakeValidationError("messages must contain at least one message")

    has_tool_blocks = False
    previous_tool_use_ids: list[str] = []
    expected_role = "user"
    for index, message in enumerate(messages):
        role = message.get("role")
        if role != expected_role:
            raise FakeValidationError(
                f"message {index}: expected role '{expected_role}' but found '{role}' (roles must alternate, starting with user)"
            )
        if not message.get("content"):
            raise FakeValidationError(f"message {index}: content must not be empty")

        results = message_tool_results(message)
        uses = message_tool_uses(message)
        if results:
            has_tool_blocks = True
            if role != "user":
                raise FakeValidationError(f"message {index}: toolResult blocks belong in a user message")
            result_ids = [r["toolUseId"] for r in results]
            if sorted(result_ids) != sorted(previous_tool_use_ids):
                raise FakeValidationError(
                    f"message {index}: toolResult IDs {sorted(result_ids)} do not match the toolUse IDs "
                    f"{sorted(previous_tool_use_ids)} of the previous assistant message"
                )
        elif previous_tool_use_ids:
            raise FakeValidationError(
                f"message {index}: the previous assistant message used tools, so this user message must answer with toolResult blocks"
            )
        if uses:
            has_tool_blocks = True
            if role != "assistant":
                raise FakeValidationError(f"message {index}: toolUse blocks belong in an assistant message")
        previous_tool_use_ids = [u["toolUseId"] for u in uses]
        expected_role = "assistant" if role == "user" else "user"

    if messages[-1].get("role") != "user":
        raise FakeValidationError("the last message must have role 'user'")
    if has_tool_blocks and not request.get("toolConfig"):
        raise FakeValidationError("toolConfig must be provided when messages contain toolUse or toolResult blocks")


_tool_counter = {"n": 0}


def _usage(input_tokens: int, output_tokens: int) -> dict:
    return {"inputTokens": input_tokens, "outputTokens": output_tokens, "totalTokens": input_tokens + output_tokens}


def text_response(text: str, input_tokens: int = 40, output_tokens: int = 12, stop_reason: str = "end_turn") -> dict:
    """Build a fake Converse response that contains only text."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop_reason,
        "usage": _usage(input_tokens, output_tokens),
        "metrics": {"latencyMs": 1},
    }


def tool_use_response(*calls: tuple[str, dict], text: str = "", input_tokens: int = 60, output_tokens: int = 25) -> dict:
    """Build a fake response where the model asks for one or more tools."""
    content: list[dict] = []
    if text:
        content.append({"text": text})
    for name, tool_input in calls:
        _tool_counter["n"] += 1
        content.append({"toolUse": {"toolUseId": f"tooluse_offline_{_tool_counter['n']}", "name": name, "input": tool_input}})
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "tool_use",
        "usage": _usage(input_tokens, output_tokens),
        "metrics": {"latencyMs": 1},
    }


class FakeBedrockRuntime:
    """Replays scripted converse() responses. Records every request in .calls.

    A script item is either a response dict or a function that takes the request
    and returns a response dict (useful when the reply depends on the request).
    """

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[dict] = []
        self.guardrail_calls: list[dict] = []
        self.guardrail_script: list[dict] = []

    def converse(self, **request: Any) -> dict:
        self.calls.append(copy.deepcopy(request))
        validate_converse_request(request)
        if not self.script:
            raise AssertionError(
                "FakeBedrockRuntime script exhausted: the lesson made more model calls than the offline script expects."
            )
        item = self.script.pop(0)
        return item(request) if callable(item) else item

    def apply_guardrail(self, **request: Any) -> dict:
        self.guardrail_calls.append(copy.deepcopy(request))
        if self.guardrail_script:
            return self.guardrail_script.pop(0)
        return {"action": "NONE", "outputs": [], "assessments": []}


# --------------------------------------------------------------------------
# A safe arithmetic evaluator (the first demo tool)
# --------------------------------------------------------------------------
_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str) -> str:
    """Evaluate plain arithmetic. No names, no calls, no attribute access."""
    if len(expression) > 200:
        raise ValueError("expression too long (max 200 characters)")
    tree = ast.parse(expression.strip(), mode="eval")

    def check_size(value: Any) -> Any:
        if isinstance(value, int) and value.bit_length() > 10_000:
            raise ValueError("result too large")
        return value

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large (max 100)")
            return check_size(_BINARY[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](evaluate(node.operand))
        raise ValueError("only numbers and + - * / // % ** ( ) are allowed")

    return str(evaluate(tree))
