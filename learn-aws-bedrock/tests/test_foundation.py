"""Tests for lab_common.py (fake client, validator, helpers) and lab_tools.py (registry, sandbox)."""
import os

import pytest

import lab_common as lc
import lab_tools


# ---------------------------------------------------------------- safe_calculate
@pytest.mark.parametrize(
    "expression, expected",
    [("17 * 23 + 4", "395"), ("(1250 * 12) / 4", "3750.0"), ("2 ** 10", "1024"), ("-5 + 3", "-2"), ("7 // 2", "3"), ("7 % 4", "3")],
)
def test_safe_calculate_valid(expression, expected):
    assert lc.safe_calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "open('x')", "a + 1", "1 + 'a'", "2 ** 1000", "(9**99)**99", "x" * 201, "[1, 2]", "True + 1", ""],
)
def test_safe_calculate_rejects_unsafe_or_huge(expression):
    with pytest.raises((ValueError, SyntaxError)):
        lc.safe_calculate(expression)


def test_safe_calculate_division_by_zero_is_an_error():
    with pytest.raises(ZeroDivisionError):
        lc.safe_calculate("1 / 0")


# ---------------------------------------------------------------- message helpers
def test_tool_result_block_shapes():
    assert lc.tool_result_block("t1", "ok") == {"toolResult": {"toolUseId": "t1", "content": [{"text": "ok"}]}}
    assert lc.tool_result_block("t1", {"a": 1})["toolResult"]["content"] == [{"json": {"a": 1}}]
    err = lc.tool_result_block("t1", "bad", is_error=True)["toolResult"]
    assert err["status"] == "error"
    # lists are not valid json content blocks, so they become text
    assert lc.tool_result_block("t1", [1, 2])["toolResult"]["content"] == [{"text": "[1, 2]"}]


def test_add_usage_and_total():
    total: dict = {}
    lc.add_usage(total, {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15})
    lc.add_usage(total, {"inputTokens": 1, "outputTokens": 2})
    lc.add_usage(total, None)
    assert total == {"inputTokens": 11, "outputTokens": 7, "totalTokens": 15}
    assert lc.total_tokens({"inputTokens": 3, "outputTokens": 4}) == 7


def test_tool_spec_shape():
    spec = lc.tool_spec("t", "d", {"x": {"type": "string"}}, ["x"])
    assert spec["toolSpec"]["inputSchema"]["json"]["required"] == ["x"]
    assert "required" not in lc.tool_spec("t", "d")["toolSpec"]["inputSchema"]["json"]


# ---------------------------------------------------------------- validator
TOOLS = {"tools": [lc.tool_spec("calculate", "d")]}


def assistant_tool_use(tool_use_id="a1"):
    return {"role": "assistant", "content": [{"toolUse": {"toolUseId": tool_use_id, "name": "calculate", "input": {}}}]}


def request(messages, with_tools=True):
    req = {"modelId": "m", "messages": messages}
    if with_tools:
        req["toolConfig"] = TOOLS
    return req


def test_validator_accepts_a_correct_tool_round_trip():
    messages = [lc.user_text("hi"), assistant_tool_use(), {"role": "user", "content": [lc.tool_result_block("a1", "1")]}]
    lc.validate_converse_request(request(messages))


def test_validator_rejects_orphan_tool_result():
    messages = [{"role": "user", "content": [lc.tool_result_block("a1", "1")]}]
    with pytest.raises(lc.FakeValidationError, match="do not match"):
        lc.validate_converse_request(request(messages))


def test_validator_rejects_unanswered_tool_use():
    messages = [lc.user_text("hi"), assistant_tool_use(), lc.user_text("and now?")]
    with pytest.raises(lc.FakeValidationError, match="must answer with toolResult"):
        lc.validate_converse_request(request(messages))


def test_validator_rejects_wrong_ids():
    messages = [lc.user_text("hi"), assistant_tool_use("a1"), {"role": "user", "content": [lc.tool_result_block("zzz", "1")]}]
    with pytest.raises(lc.FakeValidationError, match="do not match"):
        lc.validate_converse_request(request(messages))


def test_validator_rejects_role_order_problems():
    with pytest.raises(lc.FakeValidationError, match="expected role"):
        lc.validate_converse_request(request([lc.user_text("a"), lc.user_text("b")]))
    with pytest.raises(lc.FakeValidationError, match="expected role"):
        lc.validate_converse_request(request([lc.assistant_text("a")]))


def test_validator_requires_tool_config_when_tool_blocks_present():
    messages = [lc.user_text("hi"), assistant_tool_use(), {"role": "user", "content": [lc.tool_result_block("a1", "1")]}]
    with pytest.raises(lc.FakeValidationError, match="toolConfig"):
        lc.validate_converse_request(request(messages, with_tools=False))


def test_validator_requires_last_message_user_and_nonempty_content():
    with pytest.raises(lc.FakeValidationError):
        lc.validate_converse_request(request([lc.user_text("a"), lc.assistant_text("b")]))
    with pytest.raises(lc.FakeValidationError, match="empty"):
        lc.validate_converse_request(request([{"role": "user", "content": []}]))


def test_text_with_tool_results_in_same_user_message_is_allowed():
    messages = [
        lc.user_text("hi"),
        assistant_tool_use(),
        {"role": "user", "content": [lc.tool_result_block("a1", "1"), {"text": "<reminder>x</reminder>"}]},
    ]
    lc.validate_converse_request(request(messages))


# ---------------------------------------------------------------- fake client
def test_fake_client_replays_script_in_order_and_records_calls():
    client = lc.FakeBedrockRuntime([lc.text_response("one"), lc.text_response("two")])
    first = client.converse(modelId="m", messages=[lc.user_text("a")])
    second = client.converse(modelId="m", messages=[lc.user_text("b")])
    assert lc.message_text(first["output"]["message"]) == "one"
    assert lc.message_text(second["output"]["message"]) == "two"
    assert len(client.calls) == 2


def test_fake_client_script_exhausted_is_loud():
    client = lc.FakeBedrockRuntime([])
    with pytest.raises(AssertionError, match="exhausted"):
        client.converse(modelId="m", messages=[lc.user_text("a")])


def test_fake_client_supports_callable_script_items():
    client = lc.FakeBedrockRuntime([lambda req: lc.text_response("n=" + str(len(req["messages"])))])
    response = client.converse(modelId="m", messages=[lc.user_text("a")])
    assert lc.message_text(response["output"]["message"]) == "n=1"


def test_tool_use_response_generates_unique_ids():
    one = lc.tool_use_response(("a", {}), ("b", {}))
    ids = [u["toolUseId"] for u in lc.message_tool_uses(one["output"]["message"])]
    assert len(set(ids)) == 2 and one["stopReason"] == "tool_use"


def test_offline_helpers():
    assert lc.get_model_id(True) == lc.OFFLINE_MODEL_ID
    assert isinstance(lc.make_runtime_client(True, []), lc.FakeBedrockRuntime)


def test_get_model_id_live_requires_setting(monkeypatch):
    with pytest.raises(SystemExit):
        lc.get_model_id(False)
    monkeypatch.setenv("BEDROCK_MODEL_ID", " some-id ")
    assert lc.get_model_id(False) == "some-id"


# ---------------------------------------------------------------- error explanations
class FakeClientError(Exception):
    def __init__(self, code, message="boom"):
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}}


@pytest.mark.parametrize(
    "code, expected",
    [
        ("ExpiredTokenException", "expired"),
        ("AccessDeniedException", "Access denied"),
        ("ValidationException", "rejected"),
        ("ThrottlingException", "limit"),
        ("ResourceNotFoundException", "not found"),
    ],
)
def test_explain_aws_error_maps_codes(code, expected):
    assert expected.lower() in lc.explain_aws_error(FakeClientError(code)).lower()


def test_explain_aws_error_by_class_name_and_fallback():
    class NoRegionError(Exception):
        pass

    class NoCredentialsError(Exception):
        pass

    assert "region" in lc.explain_aws_error(NoRegionError()).lower()
    assert "credentials" in lc.explain_aws_error(NoCredentialsError()).lower()
    assert "RuntimeError" in lc.explain_aws_error(RuntimeError("weird"))


# ---------------------------------------------------------------- registry
def make_registry():
    registry = lab_tools.ToolRegistry()
    registry.register("echo", "Echo text", {"text": {"type": "string"}}, ["text"], lambda a: a["text"])
    registry.register("boom", "Always fails", {}, [], lambda a: 1 / 0)
    return registry


def test_registry_unknown_tool_and_missing_args_are_error_results():
    registry = make_registry()
    output, is_error = registry.call("nope", {})
    assert is_error and "Unknown tool" in output and "echo" in output
    output, is_error = registry.call("echo", {})
    assert is_error and "text" in output
    output, is_error = registry.call("echo", "not a dict")
    assert is_error


def test_registry_catches_handler_exceptions():
    output, is_error = make_registry().call("boom", {})
    assert is_error and "ZeroDivisionError" in output


def test_registry_success_and_specs_and_subset():
    registry = make_registry()
    assert registry.call("echo", {"text": "hi"}) == ("hi", False)
    assert [s["toolSpec"]["name"] for s in registry.specs()] == ["echo", "boom"]
    sub = registry.subset(["echo"])
    assert sub.names() == ["echo"]
    assert sub.call("boom", {})[1] is True  # not available in the subset


def test_registry_rejects_duplicate_names():
    registry = make_registry()
    with pytest.raises(ValueError):
        registry.register("echo", "again", {}, [], lambda a: "")


# ---------------------------------------------------------------- sandbox
def test_sandbox_blocks_escape_and_absolute_paths(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    with pytest.raises(lab_tools.SandboxError):
        lab_tools.resolve_in_sandbox(root, "../outside.txt")
    with pytest.raises(lab_tools.SandboxError):
        lab_tools.resolve_in_sandbox(root, str(tmp_path / "outside.txt"))
    assert lab_tools.resolve_in_sandbox(root, "sub/file.txt") == (root / "sub" / "file.txt").resolve()
    assert lab_tools.resolve_in_sandbox(root, "") == root.resolve()


def test_sandbox_blocks_symlink_escape(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope")
    try:
        os.symlink(outside, root / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available on this system")
    with pytest.raises(lab_tools.SandboxError):
        lab_tools.resolve_in_sandbox(root, "link/secret.txt")


def test_default_registry_file_tools(tmp_path):
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("hello", encoding="utf-8")
    (root / "big.txt").write_text("x" * (lab_tools.MAX_FILE_CHARS + 50), encoding="utf-8")
    registry = lab_tools.build_default_registry(root)
    assert registry.call("calculate", {"expression": "2+2"}) == ("4", False)
    listing, err = registry.call("list_dir", {})
    assert not err and listing.splitlines() == ["sub/", "a.txt", "big.txt"]
    assert registry.call("read_text_file", {"path": "a.txt"}) == ("hello", False)
    big, err = registry.call("read_text_file", {"path": "big.txt"})
    assert not err and "truncated" in big
    assert registry.call("read_text_file", {"path": "missing.txt"})[1] is True
    assert registry.call("read_text_file", {"path": "../x"})[1] is True
    assert registry.call("list_dir", {"path": "a.txt"})[1] is True
