"""Bedrock access for rakaRAG's live proof of concept, with a hard token cap and an offline fake.

  TokenBudget      counts every token a run spends and refuses any call that could push it over the cap
  load_config      reads model ids from the environment and refuses to guess them
  Bedrock          one place that calls Converse (text) and the embedding model, with budget checks
  OfflineBedrock   a fake bedrock-runtime that checks request shapes and answers with the r03 fakes,
                   so the whole live code path runs without an account and without cost

Request and response shapes follow the AWS documentation as I know it (Converse; Titan Text Embeddings V2
and Cohere Embed through InvokeModel). They have been exercised only against the offline fake, so the first
live call may need a small fix: that is what `--live --smoke` is for. Pure standard library; boto3 is
imported only when a live client is made.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import raka_common as rc

ROOT = rc.ROOT
DEFAULT_CAP = 20_000
HARD_CEILING = 50_000           # a deliberate code change is needed to go above this
RESERVE_TOKENS_PER_WORD = 1.6   # deliberately pessimistic: the guard assumes the worst for a single call
ACTUAL_TOKENS_PER_WORD = 1.3    # what the offline fake reports


class ConfigError(Exception):
    """Something needed for a live run is missing or unsafe."""


class BudgetExceeded(Exception):
    def __init__(self, stage: str, label: str, needed: int, remaining: int):
        super().__init__(f"{stage}/{label}: this call could use up to {needed} tokens but only {remaining} remain under the cap")
        self.stage, self.label, self.needed, self.remaining = stage, label, needed, remaining


class LiveCallError(Exception):
    """A call to Bedrock failed; the message says what to try."""


def words(text: str) -> int:
    return len((text or "").split())


def reserve_tokens(prompt_words: int, max_output_tokens: int = 0) -> int:
    """Worst-case tokens for one call: pessimistic input estimate plus the whole output allowance."""
    return math.ceil(prompt_words * RESERVE_TOKENS_PER_WORD) + max_output_tokens


# ---------------------------------------------------------------- the budget
@dataclass
class TokenBudget:
    cap: int
    used_in: int = 0
    used_out: int = 0
    calls: list = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.cap, int) or self.cap <= 0:
            raise ConfigError("the token cap must be a positive whole number")
        if self.cap > HARD_CEILING:
            raise ConfigError(f"a cap of {self.cap} is above the built-in ceiling of {HARD_CEILING}; change HARD_CEILING in raka_bedrock.py on purpose if you really mean it")

    @property
    def used(self) -> int:
        return self.used_in + self.used_out

    @property
    def remaining(self) -> int:
        return self.cap - self.used

    def check(self, worst_case: int, stage: str, label: str) -> None:
        """Refuse before the call if even its worst case would not fit."""
        if self.used + worst_case > self.cap:
            raise BudgetExceeded(stage, label, worst_case, self.remaining)

    def record(self, stage: str, kind: str, tokens_in: int, tokens_out: int, label: str, estimated: bool = False) -> None:
        self.used_in += tokens_in
        self.used_out += tokens_out
        self.calls.append({"stage": stage, "kind": kind, "label": label, "in": tokens_in, "out": tokens_out, "estimated": estimated})

    def by_stage(self) -> dict:
        out: dict = {}
        for c in self.calls:
            s = out.setdefault(c["stage"], {"calls": 0, "in": 0, "out": 0, "total": 0})
            s["calls"] += 1
            s["in"] += c["in"]
            s["out"] += c["out"]
            s["total"] += c["in"] + c["out"]
        return out

    def to_dict(self) -> dict:
        return {"cap": self.cap, "used": self.used, "used_in": self.used_in, "used_out": self.used_out,
                "remaining": self.remaining, "calls": len(self.calls), "by_stage": self.by_stage()}


# ---------------------------------------------------------------- configuration
@dataclass(frozen=True)
class LiveConfig:
    extract_model: str
    query_model: str
    embed_model: str
    region: str
    profile: str
    cap: int
    embed_dims: int
    offline: bool = False

    @property
    def embed_family(self) -> str:
        m = self.embed_model.lower()
        if "titan-embed" in m or m.startswith("offline"):
            return "titan"
        if "cohere.embed" in m:
            return "cohere"
        raise ConfigError(f"embedding model {self.embed_model!r} is not one this code knows how to call (Titan Text Embeddings or Cohere Embed)")


def load_env() -> None:
    """Load .env from the repo root if python-dotenv is installed. Never prints values."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def load_config(env: dict | None = None, offline: bool = False) -> LiveConfig:
    env = dict(os.environ if env is None else env)
    cap_text = (env.get("RAKA_MAX_TOTAL_TOKENS") or str(DEFAULT_CAP)).strip()
    try:
        cap = int(cap_text)
    except ValueError:
        raise ConfigError(f"RAKA_MAX_TOTAL_TOKENS must be a whole number, got {cap_text!r}") from None
    if cap > HARD_CEILING or cap <= 0:
        TokenBudget(cap)  # raises the right ConfigError
    if offline:
        return LiveConfig("offline-extract", "offline-query", "offline-titan-embed", "offline", "", cap, 256, True)
    problems = []
    if (env.get("RAKA_PROVIDER") or "").strip().lower() != "bedrock":
        problems.append("RAKA_PROVIDER must be 'bedrock'")
    for name in ("RAKA_EXTRACT_MODEL", "RAKA_QUERY_MODEL", "RAKA_EMBED_MODEL"):
        if not (env.get(name) or "").strip():
            problems.append(f"{name} is empty (paste an id from your own account; run learn-aws-bedrock/b00_setup to list them)")
    region = (env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "").strip()
    if not region:
        problems.append("AWS_REGION is empty")
    try:
        dims = int((env.get("RAKA_EMBED_DIMS") or "256").strip())
    except ValueError:
        dims = 0
    if dims not in (256, 512, 1024):
        problems.append("RAKA_EMBED_DIMS must be 256, 512 or 1024")
    if problems:
        raise ConfigError("cannot start a live run:\n  - " + "\n  - ".join(problems))
    config = LiveConfig(env["RAKA_EXTRACT_MODEL"].strip(), env["RAKA_QUERY_MODEL"].strip(), env["RAKA_EMBED_MODEL"].strip(),
                        region, (env.get("AWS_PROFILE") or "").strip(), cap, dims)
    config.embed_family  # raises if the embedding model family is unsupported
    return config


def explain_aws_error(exc: Exception) -> str:
    """Turn common AWS errors into a hint a person can act on (same idea as the Bedrock course)."""
    name = type(exc).__name__
    response = getattr(exc, "response", None)
    code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
    text = str(exc)
    low = text.lower()
    if "expired" in low or code in ("ExpiredTokenException", "ExpiredToken") or name == "TokenRetrievalError":
        return "Your AWS login expired. Sign in again the way you normally do (SSO), then retry."
    if name in ("NoCredentialsError", "ProfileNotFound") or code == "UnrecognizedClientException":
        return "AWS credentials were not found or are not valid. Check AWS_PROFILE and that you are signed in."
    if name == "NoRegionError":
        return "No AWS region is set. Put AWS_REGION in .env."
    if code == "AccessDeniedException":
        return "Access denied. Check that model access is enabled for this model in this Region and that your role allows bedrock:InvokeModel on it."
    if code == "ValidationException":
        return ("Bedrock rejected the request. Common causes: a wrong model id, a model that needs an inference profile id, "
                "a model not offered in this Region, or a setting this model does not accept. " + text[:200])
    if code in ("ThrottlingException", "ServiceQuotaExceededException"):
        return "Rate or quota limit reached. Wait a moment and retry."
    if code in ("ResourceNotFoundException", "ModelNotReadyException"):
        return "The model was not found or is not ready in this Region."
    return f"{name}: {text[:300]}"


def make_client(config: LiveConfig):
    """A bedrock-runtime client: the offline fake, or a real boto3 client (boto3 imported only here)."""
    if config.offline:
        return OfflineBedrock(config.embed_dims)
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ConfigError("boto3 is not installed. Run: pip install boto3") from None
    session = boto3.Session(profile_name=config.profile or None, region_name=config.region)
    return session.client("bedrock-runtime", config=Config(retries={"max_attempts": 2, "mode": "standard"}, read_timeout=60))


# ---------------------------------------------------------------- calls, always through the budget
def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class Bedrock:
    """The only place that spends tokens. Every call is checked against the budget first and recorded after."""

    def __init__(self, client, config: LiveConfig, budget: TokenBudget):
        self.client, self.config, self.budget = client, config, budget

    def converse(self, model_id: str, prompt: str, max_tokens: int, stage: str, label: str, system: str | None = None) -> str:
        self.budget.check(reserve_tokens(words(prompt) + words(system or ""), max_tokens), stage, label)
        request = {"modelId": model_id, "messages": [{"role": "user", "content": [{"text": prompt}]}],
                   "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0}}
        if system:
            request["system"] = [{"text": system}]
        try:
            response = self.client.converse(**request)
        except Exception as exc:  # noqa: BLE001 - re-raised with a human hint
            raise LiveCallError(explain_aws_error(exc)) from exc
        usage = response.get("usage") or {}
        if usage:
            tokens_in, tokens_out, estimated = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0)), False
        else:  # no usage block: count the worst case rather than nothing
            tokens_in, tokens_out, estimated = reserve_tokens(words(prompt)), max_tokens, True
        self.budget.record(stage, "converse", tokens_in, tokens_out, label, estimated)
        try:
            return "".join(block.get("text", "") for block in response["output"]["message"]["content"])
        except (KeyError, TypeError):
            raise LiveCallError("unexpected Converse response shape (no output.message.content)") from None

    def embed(self, text: str, input_type: str, stage: str, label: str) -> list[float]:
        """input_type: 'document' or 'query'. Returns a unit-length vector."""
        self.budget.check(reserve_tokens(words(text)), stage, label)
        family = self.config.embed_family
        if family == "titan":
            body = {"inputText": text, "dimensions": self.config.embed_dims, "normalize": True}
        else:
            body = {"texts": [text], "input_type": "search_document" if input_type == "document" else "search_query"}
        try:
            response = self.client.invoke_model(modelId=self.config.embed_model, body=json.dumps(body),
                                                contentType="application/json", accept="application/json")
            payload = json.loads(response["body"].read())
        except Exception as exc:  # noqa: BLE001
            raise LiveCallError(explain_aws_error(exc)) from exc
        try:
            if family == "titan":
                vector, counted = payload["embedding"], payload.get("inputTextTokenCount")
            else:
                embeddings = payload["embeddings"]
                vector, counted = (embeddings["float"][0] if isinstance(embeddings, dict) else embeddings[0]), None
        except (KeyError, IndexError, TypeError):
            raise LiveCallError("unexpected embedding response shape") from None
        if not vector or not all(isinstance(x, (int, float)) for x in vector):
            raise LiveCallError("the embedding came back empty or not numeric")
        estimated = counted is None
        self.budget.record(stage, "embed", int(counted) if counted is not None else reserve_tokens(words(text)), 0, label, estimated)
        return _unit([float(x) for x in vector])


# ---------------------------------------------------------------- an offline fake of bedrock-runtime
class OfflineBedrock:
    """Fake bedrock-runtime. Rejects malformed requests the way the real service would, then answers using the
    r03 fakes. Records every request in .calls (prompts are kept in memory only)."""

    def __init__(self, dims: int = 256):
        from raka_fake_model import LEXICON_V2, FakeExtractor, FakeKeywordModel
        self.dims = dims
        self.calls: list[dict] = []
        self._extract = FakeExtractor(lexicon=LEXICON_V2)
        self._keywords = FakeKeywordModel(lexicon=LEXICON_V2)

    def converse(self, **request) -> dict:
        self.calls.append({"api": "converse", "request": request})
        messages = request.get("messages")
        if not request.get("modelId"):
            raise ValueError("modelId is required")
        if not (isinstance(messages, list) and len(messages) == 1 and messages[0].get("role") == "user"):
            raise ValueError("this fake accepts exactly one user message")
        content = messages[0].get("content")
        if not (isinstance(content, list) and content and isinstance(content[0].get("text"), str) and content[0]["text"].strip()):
            raise ValueError("message content must be a non-empty text block")
        config = request.get("inferenceConfig") or {}
        if not isinstance(config.get("maxTokens"), int) or config["maxTokens"] <= 0:
            raise ValueError("inferenceConfig.maxTokens must be a positive integer")
        prompt = content[0]["text"]
        if "You extract a knowledge graph" in prompt:
            text = self._extract(prompt)
        elif "You extract search keywords" in prompt:
            text = self._keywords(prompt)
        else:
            ids = re.findall(r"^\[([0-9]{2}_[a-z0-9_]+)(?:#\d+)?\]", prompt, re.M)
            text = "Offline answer drawn from " + " and ".join(f"[{i}]" for i in ids[:2]) + "." if ids else "The sources do not contain the answer."
        tokens_in = math.ceil(words(prompt) * ACTUAL_TOKENS_PER_WORD)
        tokens_out = min(config["maxTokens"], math.ceil(words(text) * ACTUAL_TOKENS_PER_WORD))
        return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "stopReason": "end_turn",
                "usage": {"inputTokens": tokens_in, "outputTokens": tokens_out, "totalTokens": tokens_in + tokens_out}}

    def invoke_model(self, **request) -> dict:
        self.calls.append({"api": "invoke_model", "request": request})
        if not request.get("modelId"):
            raise ValueError("modelId is required")
        body = json.loads(request["body"])
        if "inputText" in body:  # Titan Text Embeddings shape
            text, dims = body["inputText"], int(body.get("dimensions", self.dims))
            if dims not in (256, 512, 1024) or not isinstance(text, str) or not text.strip():
                raise ValueError("bad Titan embedding request")
            payload = {"embedding": rc.hash_embed(text, dims), "inputTextTokenCount": math.ceil(words(text) * ACTUAL_TOKENS_PER_WORD)}
        elif "texts" in body:  # Cohere Embed shape
            if body.get("input_type") not in ("search_document", "search_query") or not body["texts"]:
                raise ValueError("bad Cohere embedding request")
            payload = {"embeddings": [rc.hash_embed(t, self.dims) for t in body["texts"]]}
        else:
            raise ValueError("unrecognised embedding request body")
        return {"body": io.BytesIO(json.dumps(payload).encode("utf-8"))}
