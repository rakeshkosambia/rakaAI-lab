"""Behaviour tests for chapters b00 .. b08, all offline."""
import json

import pytest

import lab_common as lc
import lab_tools

CHAPTERS = [
    "b00_setup",
    "b01_converse_loop",
    "b02_tool_use",
    "b03_permission",
    "b04_hooks",
    "b05_todo_plan",
    "b06_subagent",
    "b07_skill_loading",
    "b08_context_compact",
]


def error_results(messages):
    found = []
    for message in messages:
        for result in lc.message_tool_results(message):
            if result.get("status") == "error":
                found.append("".join(c.get("text", "") for c in result["content"]))
    return found


# ------------------------------------------------------------------ every chapter runs offline
@pytest.mark.parametrize("folder", CHAPTERS)
def test_every_chapter_runs_offline(chapter, folder, capsys, tmp_path, monkeypatch):
    module = chapter(folder)
    if folder == "b04_hooks":
        monkeypatch.setattr(lc, "ROOT", tmp_path)  # audit log goes to tmp
    assert module.main(["--offline"]) == 0
    assert capsys.readouterr().out.strip()


# ------------------------------------------------------------------ b01
def test_b01_loop_computes_and_keeps_history(chapter):
    b01 = chapter("b01_converse_loop")
    client = lc.FakeBedrockRuntime(b01.offline_script())
    messages = [lc.user_text("go")]
    text, stop, usage = b01.agent_loop(client, "m", messages, verbose=False)
    assert "395" in text and stop == "end_turn"
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert usage["totalTokens"] == 85 + 52


def test_b01_tool_failure_becomes_error_result_and_loop_continues(chapter):
    b01 = chapter("b01_converse_loop")
    client = lc.FakeBedrockRuntime([lc.tool_use_response(("calculate", {"expression": "1/0"})), lc.text_response("cannot divide by zero")])
    messages = [lc.user_text("go")]
    text, stop, _ = b01.agent_loop(client, "m", messages, verbose=False)
    assert stop == "end_turn"
    assert any("ZeroDivisionError" in e for e in error_results(messages))


def test_b01_unknown_tool_is_an_error_result(chapter):
    b01 = chapter("b01_converse_loop")
    client = lc.FakeBedrockRuntime([lc.tool_use_response(("rm_rf", {})), lc.text_response("ok")])
    messages = [lc.user_text("go")]
    b01.agent_loop(client, "m", messages, verbose=False)
    assert any("unknown tool" in e for e in error_results(messages))


def test_b01_max_turns_stops_a_runaway_loop(chapter):
    b01 = chapter("b01_converse_loop")
    script = [lc.tool_use_response(("calculate", {"expression": "1+1"})) for _ in range(5)]
    client = lc.FakeBedrockRuntime(script)
    text, stop, _ = b01.agent_loop(client, "m", [lc.user_text("go")], max_turns=3, verbose=False)
    assert stop == "max_turns" and len(client.calls) == 3


def test_b01_max_tokens_is_returned_not_executed(chapter):
    b01 = chapter("b01_converse_loop")
    client = lc.FakeBedrockRuntime([lc.text_response("partial", stop_reason="max_tokens")])
    text, stop, _ = b01.agent_loop(client, "m", [lc.user_text("go")], verbose=False)
    assert stop == "max_tokens" and text == "partial"


# ------------------------------------------------------------------ b02
def test_b02_sandbox_escape_is_refused_and_reported(chapter, tmp_path):
    b02 = chapter("b02_tool_use")
    root = tmp_path / "ws"
    root.mkdir()
    (root / "notes.txt").write_text("Q1 1250", encoding="utf-8")
    registry = lab_tools.build_default_registry(root)
    client = lc.FakeBedrockRuntime(b02.offline_script())
    messages = [lc.user_text("go")]
    text, stop, _ = b02.agent_loop(client, "m", messages, registry, verbose=False)
    assert stop == "end_turn"
    assert any("sandbox" in e for e in error_results(messages))


# ------------------------------------------------------------------ b03
class FakeTool:
    def __init__(self, risk):
        self.risk = risk


def test_b03_policy_modes(chapter):
    b03 = chapter("b03_permission")
    read, write = FakeTool("read"), FakeTool("write")
    assert b03.PermissionPolicy("readonly").check(write, {"path": "a.txt"})[0] == "deny"
    assert b03.PermissionPolicy("default").check(write, {"path": "a.txt"})[0] == "ask"
    assert b03.PermissionPolicy("auto").check(write, {"path": "a.txt"})[0] == "allow"
    for mode in ("readonly", "default", "auto"):
        assert b03.PermissionPolicy(mode).check(read, {"path": "a.txt"})[0] == "allow"
    with pytest.raises(ValueError):
        b03.PermissionPolicy("yolo")


@pytest.mark.parametrize("name", [".env", ".env.local", "server.pem", "api.key", "my_credentials.json", "client_secret.json", "id_rsa", "id_rsa.pub", "SECRETS.TXT"])
def test_b03_deny_patterns_beat_allow_even_in_auto_mode(chapter, name):
    b03 = chapter("b03_permission")
    for risk in ("read", "write"):
        assert b03.PermissionPolicy("auto").check(FakeTool(risk), {"path": f"sub/{name}"})[0] == "deny"


def test_b03_loop_denies_protected_file_and_tells_the_model_why(chapter, tmp_path):
    b03 = chapter("b03_permission")
    root = tmp_path / "ws"
    root.mkdir()
    (root / ".env").write_text("SECRET=1", encoding="utf-8")
    registry = lab_tools.build_default_registry(root)
    b03.add_write_tool(registry, root)
    client = lc.FakeBedrockRuntime([lc.tool_use_response(("read_text_file", {"path": ".env"})), lc.text_response("refused")])
    messages = [lc.user_text("go")]
    b03.agent_loop(client, "m", messages, registry, b03.PermissionPolicy("auto"), verbose=False)
    assert any("PERMISSION DENIED" in e for e in error_results(messages))
    assert "SECRET=1" not in json.dumps(messages)


def test_b03_write_needs_approval_in_default_mode(chapter, tmp_path):
    b03 = chapter("b03_permission")
    root = tmp_path / "ws"
    root.mkdir()
    call = ("write_text_file", {"path": "x.txt", "content": "hello"})

    registry = lab_tools.build_default_registry(root)
    b03.add_write_tool(registry, root)
    messages = [lc.user_text("go")]
    b03.agent_loop(lc.FakeBedrockRuntime([lc.tool_use_response(call), lc.text_response("done")]), "m", messages, registry, b03.PermissionPolicy("default"), approve_all=False, verbose=False)
    assert not (root / "out" / "x.txt").exists()  # no console in tests -> not approved
    assert any("did not approve" in e for e in error_results(messages))

    registry = lab_tools.build_default_registry(root)
    b03.add_write_tool(registry, root)
    messages = [lc.user_text("go")]
    b03.agent_loop(lc.FakeBedrockRuntime([lc.tool_use_response(call), lc.text_response("done")]), "m", messages, registry, b03.PermissionPolicy("default"), approve_all=True, verbose=False)
    assert (root / "out" / "x.txt").read_text(encoding="utf-8") == "hello"


def test_b03_readonly_mode_never_writes(chapter, tmp_path):
    b03 = chapter("b03_permission")
    root = tmp_path / "ws"
    root.mkdir()
    registry = lab_tools.build_default_registry(root)
    b03.add_write_tool(registry, root)
    messages = [lc.user_text("go")]
    script = [lc.tool_use_response(("write_text_file", {"path": "x.txt", "content": "hi"})), lc.text_response("done")]
    b03.agent_loop(lc.FakeBedrockRuntime(script), "m", messages, registry, b03.PermissionPolicy("readonly"), approve_all=True, verbose=False)
    assert not (root / "out").exists()


def test_b03_write_tool_cannot_escape_out_folder(chapter, tmp_path):
    b03 = chapter("b03_permission")
    root = tmp_path / "ws"
    root.mkdir()
    registry = lab_tools.build_default_registry(root)
    b03.add_write_tool(registry, root)
    output, is_error = registry.call("write_text_file", {"path": "../escape.txt", "content": "x"})
    assert is_error and not (root / "escape.txt").exists()


# ------------------------------------------------------------------ b04
def test_b04_redaction(chapter):
    b04 = chapter("b04_hooks")
    aws_key = "AK" + "IA" + "A" * 16
    anthropic_key = "sk-" + "ant-" + "abcdefghijklmnop"
    text = f"id={aws_key}\nkey: {anthropic_key}\npassword = hunter2\naws_secret_access_key = EXAMPLE\nregion=ap-southeast-1"
    cleaned = b04.redact(text)
    for leaked in (aws_key, anthropic_key, "hunter2", "EXAMPLE"):
        assert leaked not in cleaned
    assert "region=ap-southeast-1" in cleaned


def test_b04_demo_secret_never_reaches_the_conversation(chapter, tmp_path):
    b04 = chapter("b04_hooks")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    b04.add_demo_config_tool(registry)
    hooks, _ = b04.build_hooks(tmp_path / "audit.jsonl", 50_000)
    messages = [lc.user_text("go")]
    b04.agent_loop(lc.FakeBedrockRuntime(b04.offline_script()), "m", messages, registry, hooks)
    assert "EXAMPLE-NOT-A-REAL-SECRET" not in json.dumps(messages)
    assert "[REDACTED]" in json.dumps(messages)


def test_b04_audit_log_is_valid_jsonl_without_raw_secrets(chapter, tmp_path):
    b04 = chapter("b04_hooks")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    b04.add_demo_config_tool(registry)
    log = tmp_path / "logs" / "audit.jsonl"
    hooks, _ = b04.build_hooks(log, 50_000)
    b04.agent_loop(lc.FakeBedrockRuntime(b04.offline_script()), "m", [lc.user_text("go")], registry, hooks)
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    events = {r["event"] for r in records}
    assert {"model_call", "model_reply", "tool_call", "tool_result"} <= events
    assert "EXAMPLE-NOT-A-REAL-SECRET" not in log.read_text(encoding="utf-8")


def test_b04_token_budget_stops_the_run(chapter, tmp_path):
    b04 = chapter("b04_hooks")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    hooks, _ = b04.build_hooks(tmp_path / "audit.jsonl", 100)  # first reply alone uses 85, second call would exceed
    script = [lc.tool_use_response(("calculate", {"expression": "1+1"}), input_tokens=90, output_tokens=20), lc.text_response("never reached")]
    client = lc.FakeBedrockRuntime(script)
    text, stop, _ = b04.agent_loop(client, "m", [lc.user_text("go")], registry, hooks)
    assert stop == "budget" and len(client.calls) == 1


def test_b04_before_tool_hook_can_block(chapter, tmp_path):
    b04 = chapter("b04_hooks")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    hooks = b04.HookRunner()
    hooks.add("before_tool", lambda ctx, name, inp: "no powers allowed" if "**" in json.dumps(inp) else None)
    messages = [lc.user_text("go")]
    script = [lc.tool_use_response(("calculate", {"expression": "2 ** 3"})), lc.text_response("blocked")]
    b04.agent_loop(lc.FakeBedrockRuntime(script), "m", messages, registry, hooks)
    assert any("BLOCKED BY HOOK: no powers allowed" in e for e in error_results(messages))


def test_b04_guardrail_blocks_and_passes(chapter, tmp_path):
    b04 = chapter("b04_hooks")
    registry = lab_tools.build_default_registry(tmp_path / "ws")

    blocked_client = lc.FakeBedrockRuntime([lc.text_response("never reached")])
    blocked_client.guardrail_script = [{"action": "GUARDRAIL_INTERVENED"}]
    hooks = b04.HookRunner()
    b04.GuardrailCheck("gr-123", "1").install(hooks)
    text, stop, _ = b04.agent_loop(blocked_client, "m", [lc.user_text("bad input")], registry, hooks)
    assert stop == "guardrail" and not blocked_client.calls
    assert blocked_client.guardrail_calls[0]["source"] == "INPUT"
    assert blocked_client.guardrail_calls[0]["guardrailIdentifier"] == "gr-123"

    ok_client = lc.FakeBedrockRuntime([lc.text_response("fine")])
    hooks = b04.HookRunner()
    b04.GuardrailCheck("gr-123").install(hooks)
    text, stop, _ = b04.agent_loop(ok_client, "m", [lc.user_text("hello")], registry, hooks)
    assert stop == "end_turn" and text == "fine"


def test_b04_unknown_hook_event_is_rejected(chapter):
    with pytest.raises(ValueError):
        chapter("b04_hooks").HookRunner().add("on_lunch", lambda: None)


def test_b04_cost_estimate_only_with_prices(chapter):
    b04 = chapter("b04_hooks")
    assert b04.TokenBudget(100).estimate_cost({"inputTokens": 1000, "outputTokens": 1000}) is None
    assert b04.TokenBudget(100, 1.0, 2.0).estimate_cost({"inputTokens": 1000, "outputTokens": 500}) == pytest.approx(2.0)


# ------------------------------------------------------------------ b05
def test_b05_todo_validation(chapter):
    b05 = chapter("b05_todo_plan")
    todos = b05.TodoList()
    with pytest.raises(ValueError, match="only one"):
        todos.update([{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}])
    with pytest.raises(ValueError, match="invalid status"):
        todos.update([{"content": "a", "status": "doing"}])
    with pytest.raises(ValueError, match="empty"):
        todos.update([{"content": " ", "status": "pending"}])
    with pytest.raises(ValueError):
        todos.update([])
    with pytest.raises(ValueError, match="too many"):
        todos.update([{"content": str(i), "status": "pending"} for i in range(21)])
    rendered = todos.update([{"content": "a", "status": "completed"}, {"content": "b", "status": "in_progress"}, {"content": "c", "status": "pending"}])
    assert rendered.splitlines() == ["[x] a", "[>] b", "[ ] c"]
    assert todos.has_open()


def test_b05_nag_added_after_three_turns_without_updating_the_plan(chapter, tmp_path):
    b05 = chapter("b05_todo_plan")
    todos = b05.TodoList()
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    b05.add_todo_tool(registry, todos)
    messages = [lc.user_text("go")]
    b05.agent_loop(lc.FakeBedrockRuntime(b05.offline_script()), "m", messages, registry, todos, verbose=False)
    reminders = [i for i, m in enumerate(messages) if m["role"] == "user" and "<reminder>" in lc.message_text(m)]
    assert len(reminders) == 1
    # the reminder follows the 3rd tool turn after the plan was written: message index 8 (0 user, 1 plan, 2 res, 3,4, 5,6, 7 asst, 8 results+reminder)
    assert reminders[0] == 8
    assert not todos.has_open()


def test_b05_no_nag_when_everything_is_completed(chapter, tmp_path):
    b05 = chapter("b05_todo_plan")
    todos = b05.TodoList()
    todos.update([{"content": "a", "status": "completed"}])
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    b05.add_todo_tool(registry, todos)
    script = [lc.tool_use_response(("calculate", {"expression": "1+1"})) for _ in range(4)] + [lc.text_response("done")]
    messages = [lc.user_text("go")]
    b05.agent_loop(lc.FakeBedrockRuntime(script), "m", messages, registry, todos, verbose=False)
    assert not any("<reminder>" in lc.message_text(m) for m in messages)


# ------------------------------------------------------------------ b06
def test_b06_parent_context_stays_clean_and_usage_rolls_up(chapter, tmp_path):
    b06 = chapter("b06_subagent")
    root = tmp_path / "ws"
    root.mkdir()
    (root / "notes.txt").write_text("SECRET-FILE-CONTENT " * 50, encoding="utf-8")
    registry = lab_tools.build_default_registry(root)
    usage: dict = {}
    client = lc.FakeBedrockRuntime(b06.offline_script())
    b06.add_task_tool(registry, client, "m", usage, verbose=False)
    messages = [lc.user_text("go")]
    text, stop = b06.run_loop(client, "m", messages, registry, b06.PARENT_SYSTEM, usage, 8, "main", verbose=False)
    assert stop == "end_turn"
    assert len(messages) == 4  # user, assistant(task), user(result), assistant(final)
    assert "SECRET-FILE-CONTENT" not in json.dumps(messages)  # file content stayed in the child's context
    assert usage["totalTokens"] == (60 + 25) * 2 + (40 + 12) * 2  # parent + child calls in one total


def test_b06_child_cannot_spawn_children(chapter, tmp_path):
    b06 = chapter("b06_subagent")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    client = lc.FakeBedrockRuntime([])
    b06.add_task_tool(registry, client, "m", {}, verbose=False)
    assert "task" in registry.names()
    child = registry.subset(b06.CHILD_TOOLS)
    assert "task" not in child.names()


def test_b06_empty_prompt_is_an_error_result_and_report_is_capped(chapter, tmp_path):
    b06 = chapter("b06_subagent")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    client = lc.FakeBedrockRuntime([lc.text_response("x" * 10_000)])
    b06.add_task_tool(registry, client, "m", {}, verbose=False)
    assert registry.call("task", {"prompt": "  "})[1] is True
    output, is_error = registry.call("task", {"prompt": "do it"})
    assert not is_error and len(output) <= b06.MAX_REPORT_CHARS


def test_b06_abnormal_child_ending_is_reported(chapter, tmp_path):
    b06 = chapter("b06_subagent")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    client = lc.FakeBedrockRuntime([lc.text_response("cut off", stop_reason="max_tokens")])
    b06.add_task_tool(registry, client, "m", {}, verbose=False)
    output, _ = registry.call("task", {"prompt": "do it"})
    assert "max_tokens" in output


# ------------------------------------------------------------------ b07
SKILL_TEXT = "---\nname: demo\ndescription: A demo skill. Use for demos.\n---\n# Body\nStep 1\n"


def test_b07_frontmatter_parsing(chapter):
    b07 = chapter("b07_skill_loading")
    meta, body = b07.parse_frontmatter(SKILL_TEXT)
    assert meta == {"name": "demo", "description": "A demo skill. Use for demos."} and body.startswith("# Body")
    assert b07.parse_frontmatter("no header here") == ({}, "no header here")
    assert b07.parse_frontmatter("---\nname: x\nnever closed")[0] == {}
    meta, _ = b07.parse_frontmatter(SKILL_TEXT.replace("\n", "\r\n"))
    assert meta["name"] == "demo"


def test_b07_local_source_and_loader(chapter, tmp_path):
    b07 = chapter("b07_skill_loading")
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "SKILL.md").write_text(SKILL_TEXT, encoding="utf-8")
    loader = b07.SkillLoader(b07.LocalSkillSource(tmp_path))
    assert loader.catalog() == {"demo": "A demo skill. Use for demos."}
    assert "- demo: A demo skill" in loader.system_section()
    assert loader.load("demo").startswith('<skill name="demo">') and "Step 1" in loader.load("demo")
    with pytest.raises(ValueError, match="Available: demo"):
        loader.load("missing")
    with pytest.raises(FileNotFoundError):
        b07.LocalSkillSource(tmp_path).read("../demo")


def test_b07_s3_source_with_fake_client(chapter):
    import io

    b07 = chapter("b07_skill_loading")

    class FakePaginator:
        def paginate(self, **kwargs):
            assert kwargs["Prefix"] == "skills/" and kwargs["Delimiter"] == "/"
            return [{"CommonPrefixes": [{"Prefix": "skills/demo/"}, {"Prefix": "skills/other/"}]}]

    class FakeS3:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return FakePaginator()

        def get_object(self, Bucket, Key):
            assert Bucket == "bkt" and Key.endswith("/SKILL.md")
            return {"Body": io.BytesIO(SKILL_TEXT.encode("utf-8"))}

    source = b07.S3SkillSource(FakeS3(), "bkt", "skills")  # prefix without trailing slash is normalised
    assert source.names() == ["demo", "other"]
    assert "Step 1" in source.read("demo")
    with pytest.raises(ValueError):
        source.read("../etc")
    assert b07.SkillLoader(source).catalog()["demo"].startswith("A demo skill")


def test_b07_shipped_skills_have_valid_headers(chapter):
    b07 = chapter("b07_skill_loading")
    loader = b07.SkillLoader(b07.LocalSkillSource(b07.SKILLS_DIR))
    catalog = loader.catalog()
    assert {"change-request-impact", "upgrade-checklist"} <= set(catalog)
    assert all(desc and desc != "(no description)" for desc in catalog.values())


def test_b07_loop_loads_skill_then_answers(chapter, tmp_path):
    b07 = chapter("b07_skill_loading")
    loader = b07.SkillLoader(b07.LocalSkillSource(b07.SKILLS_DIR))
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    b07.add_skill_tool(registry, loader)
    messages = [lc.user_text("go")]
    text, stop, _ = b07.agent_loop(lc.FakeBedrockRuntime(b07.offline_script()), "m", messages, registry, [{"text": "sys"}], verbose=False)
    assert stop == "end_turn"
    assert "Change request impact assessment" in json.dumps(messages)  # body arrived as a tool result


# ------------------------------------------------------------------ b08
def make_history(b08, rounds=6, chars=1500):
    return b08.build_demo_conversation(rounds, chars)


def test_b08_micro_compact_shrinks_old_outputs_and_keeps_pairs_valid(chapter, tmp_path):
    b08 = chapter("b08_context_compact")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    history = make_history(b08)
    shrunk = b08.micro_compact(history, keep_recent=2)
    assert b08.estimate_tokens(shrunk) < b08.estimate_tokens(history)
    assert b08.is_valid_history(shrunk, registry)[0]
    # the original is not modified
    assert "x" * 1500 in json.dumps(history)
    # the two most recent tool results are untouched
    result_msgs = [m for m in shrunk if lc.message_tool_results(m)]
    assert "x" * 1500 in json.dumps(result_msgs[-1]) and "x" * 1500 in json.dumps(result_msgs[-2])
    assert "omitted" in json.dumps(result_msgs[0])


def test_b08_find_tail_start_always_lands_on_a_real_user_message(chapter):
    b08 = chapter("b08_context_compact")
    history = make_history(b08)
    for keep_last in range(1, len(history)):
        start = b08.find_tail_start(history, keep_last)
        if start is not None:
            assert start >= 1 and b08.is_real_user_message(history[start])


def test_b08_naive_cut_is_invalid_but_safe_cut_is_valid(chapter, tmp_path):
    b08 = chapter("b08_context_compact")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    history = make_history(b08)
    assert not b08.is_valid_history(history[2:], registry)[0]  # cut inside a tool pair
    start = b08.find_tail_start(history, 6)
    assert b08.is_valid_history(history[start:], registry)[0]


def test_b08_summary_compaction_is_valid_and_alternates(chapter, tmp_path):
    b08 = chapter("b08_context_compact")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    history = make_history(b08)
    client = lc.FakeBedrockRuntime([lc.text_response("Reports 1-4: 100..400")])
    compacted = b08.summarize_compact(client, "m", history, keep_last=6)
    assert compacted[0]["role"] == "user" and "Summary" in lc.message_text(compacted[0])
    assert compacted[1]["role"] == "assistant"
    assert b08.is_valid_history(compacted, registry)[0]
    assert len(compacted) < len(history)
    # the summarisation request itself must be valid and carry the old transcript
    assert "report_1.txt" in json.dumps(client.calls[0]["messages"])


def test_b08_compact_if_needed_paths(chapter):
    b08 = chapter("b08_context_compact")
    history = make_history(b08)
    client = lc.FakeBedrockRuntime([lc.text_response("summary")])
    assert b08.compact_if_needed(client, "m", history, 10_000)[1] == "none"
    assert b08.compact_if_needed(client, "m", history, 2000)[1] == "micro"
    assert b08.compact_if_needed(client, "m", history, 800)[1] == "summary"


def test_b08_micro_only_when_there_is_no_safe_cut(chapter):
    b08 = chapter("b08_context_compact")
    messages = [lc.user_text("one big task")]
    for n in range(1, 8):  # one real user message, then a long tool loop
        tid = f"t{n}"
        messages.append({"role": "assistant", "content": [{"toolUse": {"toolUseId": tid, "name": "read_text_file", "input": {}}}]})
        messages.append({"role": "user", "content": [lc.tool_result_block(tid, "y" * 2000)]})
    client = lc.FakeBedrockRuntime([])
    compacted, how = b08.compact_if_needed(client, "m", messages, 100)
    assert how == "micro-only" and not client.calls


def test_b08_loop_compacts_before_calling_the_model(chapter, tmp_path):
    b08 = chapter("b08_context_compact")
    registry = lab_tools.build_default_registry(tmp_path / "ws")
    messages = make_history(b08)
    client = lc.FakeBedrockRuntime([lc.text_response("summary of old stuff"), lc.text_response("final answer")])
    text, stop, _ = b08.agent_loop(client, "m", messages, registry, threshold_tokens=800)
    assert text == "final answer" and stop == "end_turn"
    assert "Summary" in lc.message_text(messages[0])  # history was replaced in place


# ------------------------------------------------------------------ b00
def test_b00_masks_account_id(chapter):
    b00 = chapter("b00_setup")
    assert b00.mask_account("123456789012") == "********9012"
    assert "123456789012" not in b00.mask_account("123456789012")
    assert b00.mask_account("12") == "****"


def test_b00_listing_functions_use_the_bedrock_api_shape(chapter):
    b00 = chapter("b00_setup")

    class FakeBedrock:
        def list_foundation_models(self, **kwargs):
            assert kwargs == {"byProvider": "Anthropic"}
            return {"modelSummaries": [{"modelId": "b-model", "modelName": "B", "inferenceTypesSupported": ["ON_DEMAND"]}, {"modelId": "a-model", "modelName": "A", "modelLifecycle": {"status": "ACTIVE"}}]}

        def list_inference_profiles(self, **kwargs):
            if "nextToken" not in kwargs:
                return {"inferenceProfileSummaries": [{"inferenceProfileId": "x.claude-1", "inferenceProfileName": "Claude One", "status": "ACTIVE"}, {"inferenceProfileId": "x.other", "inferenceProfileName": "Other"}], "nextToken": "n"}
            return {"inferenceProfileSummaries": [{"inferenceProfileId": "x.anthropic-2", "inferenceProfileName": "Two"}]}

    class FakeSession:
        def client(self, name):
            assert name == "bedrock"
            return FakeBedrock()

    models = b00.list_anthropic_models(FakeSession())
    assert [m["modelId"] for m in models] == ["a-model", "b-model"]
    profiles = b00.list_anthropic_profiles(FakeSession())
    assert [p["profileId"] for p in profiles] == ["x.anthropic-2", "x.claude-1"]  # paginated, filtered, sorted


def test_b00_smoke_converse_reads_the_reply(chapter):
    b00 = chapter("b00_setup")
    client = lc.FakeBedrockRuntime([lc.text_response("hi there")])
    result = b00.smoke_converse(client, "m")
    assert result["text"] == "hi there" and result["stopReason"] == "end_turn"
    assert client.calls[0]["inferenceConfig"]["maxTokens"] == 60
