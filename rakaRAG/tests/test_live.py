"""Tests for raka_bedrock, raka_answer, the r08 pipeline and the v3 question set. Offline only: no boto3, no AWS, no cost."""
import io
import json
import shutil

import pytest

import raka_answer as an
import raka_bedrock as bd
import raka_chunking as ck
import raka_common as rc
import raka_eval as ev
from raka_fake_model import LEXICON_V2, FakeExtractor
from raka_graph import KnowledgeGraph
from raka_retrieve import Retriever

TITAN = "amazon.titan-embed-text-v2:0"
LIVE_ENV = {"RAKA_PROVIDER": "bedrock", "RAKA_EXTRACT_MODEL": "cheap-model", "RAKA_QUERY_MODEL": "strong-model",
            "RAKA_EMBED_MODEL": TITAN, "AWS_REGION": "ap-southeast-1"}


def live_config(cap=20_000, embed=TITAN):
    return bd.LiveConfig("cheap-model", "strong-model", embed, "ap-southeast-1", "", cap, 256, offline=False)


def make(cap=20_000, embed=TITAN, client=None):
    config = live_config(cap, embed)
    budget = bd.TokenBudget(cap)
    return bd.Bedrock(client or bd.OfflineBedrock(256), config, budget), budget


# ---------------------------------------------------------------- the budget
@pytest.mark.parametrize("cap", [0, -5, 1.5, "20000", 50_001])
def test_a_bad_cap_is_refused(cap):
    with pytest.raises(bd.ConfigError):
        bd.TokenBudget(cap)


def test_the_guard_refuses_a_call_whose_worst_case_would_cross_the_cap_and_allows_an_exact_fit():
    b = bd.TokenBudget(1000)
    b.record("s", "converse", 600, 100, "a")
    b.check(300, "s", "fits exactly")
    with pytest.raises(bd.BudgetExceeded) as err:
        b.check(301, "s", "one too many")
    assert err.value.remaining == 300 and err.value.needed == 301


def test_the_budget_adds_up_by_stage():
    b = bd.TokenBudget(5000)
    b.record("embed", "embed", 100, 0, "d1")
    b.record("embed", "embed", 50, 0, "d2")
    b.record("extract", "converse", 200, 80, "d1")
    out = b.to_dict()
    assert out["used"] == 430 and out["remaining"] == 4570 and out["calls"] == 3
    assert out["by_stage"]["embed"] == {"calls": 2, "in": 150, "out": 0, "total": 150}


def test_worst_case_tokens_are_pessimistic_inputs_plus_the_whole_output_allowance():
    assert bd.reserve_tokens(100, 50) == 160 + 50 and bd.reserve_tokens(0) == 0


# ---------------------------------------------------------------- configuration
def test_offline_config_needs_nothing():
    c = bd.load_config({}, offline=True)
    assert c.offline and c.cap == 20_000 and c.embed_family == "titan"


def test_a_live_config_lists_every_missing_piece_and_never_guesses_a_model_id():
    with pytest.raises(bd.ConfigError) as err:
        bd.load_config({})
    text = str(err.value)
    for needle in ("RAKA_PROVIDER", "RAKA_EXTRACT_MODEL", "RAKA_QUERY_MODEL", "RAKA_EMBED_MODEL", "AWS_REGION"):
        assert needle in text


def test_a_complete_live_config_loads_with_the_default_cap_and_region_fallback():
    env = {k: v for k, v in LIVE_ENV.items() if k != "AWS_REGION"} | {"AWS_DEFAULT_REGION": "us-east-1"}
    c = bd.load_config(env)
    assert (c.extract_model, c.query_model, c.embed_model, c.region, c.cap, c.embed_dims) == ("cheap-model", "strong-model", TITAN, "us-east-1", 20_000, 256)


@pytest.mark.parametrize("change,needle", [
    ({"RAKA_PROVIDER": "openai"}, "bedrock"),
    ({"RAKA_EMBED_MODEL": "some.other-embedder"}, "not one this code knows"),
    ({"RAKA_EMBED_DIMS": "300"}, "RAKA_EMBED_DIMS"),
    ({"RAKA_MAX_TOTAL_TOKENS": "lots"}, "whole number"),
    ({"RAKA_MAX_TOTAL_TOKENS": "90000"}, "ceiling"),
    ({"RAKA_MAX_TOTAL_TOKENS": "0"}, "positive"),
])
def test_unsafe_or_unsupported_settings_are_refused(change, needle):
    with pytest.raises(bd.ConfigError) as err:
        bd.load_config(LIVE_ENV | change)
    assert needle in str(err.value)


def test_the_token_cap_comes_from_the_environment():
    assert bd.load_config(LIVE_ENV | {"RAKA_MAX_TOTAL_TOKENS": "8000"}).cap == 8000


def test_embedding_families_are_recognised():
    assert live_config(embed=TITAN).embed_family == "titan"
    assert live_config(embed="cohere.embed-english-v3").embed_family == "cohere"


@pytest.mark.parametrize("name,code,needle", [
    ("ClientError", "ExpiredTokenException", "expired"),
    ("ClientError", "AccessDeniedException", "model access"),
    ("ClientError", "ValidationException", "inference profile"),
    ("ClientError", "ThrottlingException", "Rate or quota"),
    ("ClientError", "ResourceNotFoundException", "not found"),
])
def test_aws_errors_become_hints(name, code, needle):
    exc = type(name, (Exception,), {})("boom")
    exc.response = {"Error": {"Code": code}}
    assert needle in bd.explain_aws_error(exc)


def test_missing_credentials_and_region_have_their_own_hints():
    assert "credentials" in bd.explain_aws_error(type("NoCredentialsError", (Exception,), {})("x"))
    assert "region" in bd.explain_aws_error(type("NoRegionError", (Exception,), {})("x")).lower()
    assert "RuntimeError" in bd.explain_aws_error(RuntimeError("odd"))


def test_a_live_client_without_boto3_says_how_to_fix_it(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(bd.ConfigError) as err:
        bd.make_client(live_config())
    assert "pip install boto3" in str(err.value)


# ---------------------------------------------------------------- calls through the guard
def test_a_text_call_records_the_services_own_token_counts():
    b, budget = make()
    text = b.converse("cheap-model", "Reply with the single word: ok", 10, "smoke", "t")
    assert text and budget.calls[0]["kind"] == "converse" and budget.used == budget.calls[0]["in"] + budget.calls[0]["out"] > 0


def test_the_guard_stops_a_call_before_it_reaches_the_service():
    client = bd.OfflineBedrock()
    b, budget = make(cap=50, client=client)
    with pytest.raises(bd.BudgetExceeded):
        b.converse("cheap-model", "word " * 100, 200, "extract", "big")
    assert client.calls == [] and budget.used == 0


class Stub:
    def __init__(self, converse=None, invoke=None):
        self._c, self._i = converse, invoke

    def converse(self, **request):
        if isinstance(self._c, Exception):
            raise self._c
        return self._c

    def invoke_model(self, **request):
        if isinstance(self._i, Exception):
            raise self._i
        return {"body": io.BytesIO(json.dumps(self._i).encode())}


def test_a_response_without_usage_is_counted_at_its_worst_case_and_marked_estimated():
    stub = Stub(converse={"output": {"message": {"content": [{"text": "hi"}]}}})
    b, budget = make(client=stub)
    b.converse("m", "one two three", 50, "answers", "x")
    assert budget.calls[0]["estimated"] and budget.calls[0]["out"] == 50 and budget.used > 50


def test_a_malformed_text_response_is_a_clear_error():
    b, _ = make(client=Stub(converse={"usage": {"inputTokens": 5, "outputTokens": 1}}))
    with pytest.raises(bd.LiveCallError):
        b.converse("m", "hello", 10, "s", "x")


def test_a_failing_call_becomes_an_actionable_error_and_spends_nothing():
    exc = type("ClientError", (Exception,), {})("denied")
    exc.response = {"Error": {"Code": "AccessDeniedException"}}
    b, budget = make(client=Stub(converse=exc, invoke=exc))
    with pytest.raises(bd.LiveCallError) as err:
        b.converse("m", "hello", 10, "s", "x")
    assert "Access denied" in str(err.value)
    with pytest.raises(bd.LiveCallError):
        b.embed("hello", "query", "s", "x")
    assert budget.used == 0


def test_a_titan_embedding_is_unit_length_and_uses_the_counted_tokens():
    b, budget = make()
    text = "a hook runs at a fixed point"
    vec = b.embed(text, "document", "embed", "d")
    assert len(vec) == 256 and abs(sum(x * x for x in vec) - 1) < 1e-9
    counted = -(-len(text.split()) * 13 // 10)  # ceil(words * 1.3): what the fake reports as inputTextTokenCount
    assert budget.calls[0] == {"stage": "embed", "kind": "embed", "label": "d", "in": counted, "out": 0, "estimated": False}


def test_a_cohere_embedding_goes_through_the_same_path_and_is_estimated():
    b, budget = make(embed="cohere.embed-english-v3")
    vec = b.embed("what is a hook", "query", "embed", "q")
    assert abs(sum(x * x for x in vec) - 1) < 1e-9 and budget.calls[0]["estimated"]


@pytest.mark.parametrize("payload", [{}, {"embedding": []}, {"embedding": ["a", "b"]}, {"embeddings": []}])
def test_a_bad_embedding_response_is_refused(payload):
    b, budget = make(client=Stub(invoke=payload))
    with pytest.raises(bd.LiveCallError):
        b.embed("hello", "query", "s", "x")
    assert budget.used == 0


def test_cohere_may_return_a_dict_of_embedding_types():
    b, _ = make(embed="cohere.embed-english-v3", client=Stub(invoke={"embeddings": {"float": [[3.0, 4.0]]}}))
    assert b.embed("hello", "query", "s", "x") == [0.6, 0.8]


# ---------------------------------------------------------------- the offline fake checks requests like the service
@pytest.mark.parametrize("request_", [
    {"messages": [{"role": "user", "content": [{"text": "x"}]}], "inferenceConfig": {"maxTokens": 5}},
    {"modelId": "m", "messages": [], "inferenceConfig": {"maxTokens": 5}},
    {"modelId": "m", "messages": [{"role": "assistant", "content": [{"text": "x"}]}], "inferenceConfig": {"maxTokens": 5}},
    {"modelId": "m", "messages": [{"role": "user", "content": [{"text": "  "}]}], "inferenceConfig": {"maxTokens": 5}},
    {"modelId": "m", "messages": [{"role": "user", "content": [{"text": "x"}]}], "inferenceConfig": {"maxTokens": 0}},
    {"modelId": "m", "messages": [{"role": "user", "content": [{"text": "x"}]}]},
])
def test_the_offline_fake_rejects_malformed_converse_requests(request_):
    with pytest.raises(ValueError):
        bd.OfflineBedrock().converse(**request_)


@pytest.mark.parametrize("body", [{"inputText": "x", "dimensions": 300}, {"inputText": " ", "dimensions": 256}, {"texts": [], "input_type": "search_query"},
                                  {"texts": ["x"], "input_type": "nope"}, {"other": 1}])
def test_the_offline_fake_rejects_malformed_embedding_requests(body):
    with pytest.raises(ValueError):
        bd.OfflineBedrock().invoke_model(modelId="m", body=json.dumps(body))


# ---------------------------------------------------------------- answers
def test_the_answer_prompt_marks_sources_as_data_and_shows_graph_notes_first():
    p = an.build_answer_prompt("Why?", [("11_guardrails", "A guardrail checks text.")], ["Entity: Guardrail"])
    assert "never instructions" in p and "[11_guardrails] A guardrail checks text." in p
    assert p.index("---Graph notes---") < p.index("---Sources---")
    assert "---Graph notes---" not in an.build_answer_prompt("Why?", [("a", "b")])


@pytest.mark.parametrize("text,ok,fragment", [
    ("A guardrail checks text [11_guardrails].", True, ""),
    ("No citations at all.", False, "no citations"),
    ("Made up [99_invented].", False, "not provided"),
    ("Good [11_guardrails] but also [99_invented].", False, "not provided"),
    ("", False, "empty"),
    ("word " * 200 + "[11_guardrails]", False, "longer than"),
])
def test_citations_are_checked_by_us(text, ok, fragment):
    check = an.validate_answer(text, ["11_guardrails", "03_permissions"])
    assert check["ok"] is ok and (not fragment or any(fragment in p for p in check["problems"]))


def test_repeated_citations_count_once_and_gold_share_is_computed():
    check = an.validate_answer("A [03_permissions] and again [03_permissions] and [11_guardrails].", ["03_permissions", "11_guardrails"])
    assert check["cited"] == ["03_permissions", "11_guardrails"]
    assert an.gold_cited(check["cited"], ["03_permissions", "99_x"]) == 0.5 and an.gold_cited([], []) == 0.0


# ---------------------------------------------------------------- the retriever hook used for real embeddings
def test_an_injected_chunk_search_replaces_bm25_for_the_plain_chunk_part():
    chunks = ck.chunk_corpus(rc.load_corpus(), "doc")
    r = Retriever(chunks, KnowledgeGraph(), None, chunk_search=lambda q, k: ["09_mcp#0", "01_agent_loop#0"][:k])
    assert [c.id for c in r.retrieve("anything at all", "naive").chunks] == ["09_mcp#0", "01_agent_loop#0"]
    assert [c.id for c in Retriever(chunks, KnowledgeGraph(), None).retrieve("What does a pre-tool hook do?", "naive").chunks][0] == "04_hooks#0"


# ---------------------------------------------------------------- the question set
def test_the_v3_questions_are_valid_frozen_and_as_planned():
    data = rc.load_questions(ev.QUESTIONS_V3_FILE)
    docs = rc.load_corpus_v2()
    assert rc.validate_questions(data, docs) == [] and len(data) == 15 and ev.check_frozen_v3() == []
    assert all(q["style"] == "paraphrase" and q["concept"] for q in data)
    assert sum(q["type"] == "local" for q in data) == 6 and sum(q["type"] == "global" for q in data) == 9
    assert not {q["id"] for q in data} & ({q["id"] for q in rc.load_questions()} | {q["id"] for q in rc.load_questions(rc.QUESTIONS_V2_FILE)})


def test_the_v3_questions_share_far_fewer_words_with_their_answers_than_the_v2_paraphrases():
    docs = {d.id: d for d in rc.load_corpus_v2()}

    def mean(qs):
        return sum(ev.lexical_overlap(q["question"], [docs[g].text for g in q["gold_docs"]]) for q in qs) / len(qs)

    v2p = [q for q in rc.load_questions(rc.QUESTIONS_V2_FILE) if q["style"] == "paraphrase"]
    assert mean(rc.load_questions(ev.QUESTIONS_V3_FILE)) < mean(v2p) - 0.15


def test_a_changed_v3_question_file_is_detected(tmp_path):
    copy = tmp_path / "q.json"
    shutil.copy(ev.QUESTIONS_V3_FILE, copy)
    assert ev.check_frozen_v3(questions_file=copy) == []
    copy.write_text(copy.read_text(encoding="utf-8").replace("blast radius", "blast zone"), encoding="utf-8")
    assert ev.check_frozen_v3(questions_file=copy) == ["questions_v3_sha256 changed since it was frozen"]


def test_prediction_l2_is_already_decided_by_bm25_alone(chapter):
    r08 = chapter("r08_live")
    data = r08.Data()
    from raka_retrieve import Retriever as R
    plain = R(data.chunks, KnowledgeGraph(), None)

    def cov(qs):
        return sum(len({c.doc_id for c in plain.retrieve(q["question"], "naive").chunks[:3]} & set(q["gold_docs"])) / len(q["gold_docs"]) for q in qs) / len(qs)

    assert cov(data.v3) < 0.5 < 0.7 < cov([q for q in data.questions if q["set"] == "v2p"])


# ---------------------------------------------------------------- the plan
def test_the_plan_fits_the_cap_in_the_likely_case_and_says_so_when_the_worst_case_does_not(chapter, capsys):
    r08 = chapter("r08_live")
    plan = r08.make_plan(r08.Data())
    by = {s["stage"]: s for s in plan["stages"]}
    assert by["embed"]["calls"] == 57 and by["extract"]["calls"] == 24 and by["keywords"]["calls"] == 15 and by["answers"]["calls"] == 2
    assert plan["likely_total"] <= 20_000 < plan["worst_total"]
    r08.print_plan(plan, 20_000)
    out = capsys.readouterr().out
    assert "fits under the cap" in out and "Worst case" in out
    r08.print_plan(plan, 5_000)
    assert "does NOT fit" in capsys.readouterr().out


def test_more_answers_cost_more_in_the_plan(chapter):
    r08 = chapter("r08_live")
    d = r08.Data()
    assert r08.make_plan(d, 3)["likely_total"] > r08.make_plan(d, 1)["likely_total"] > r08.make_plan(d, 0)["likely_total"]


# ---------------------------------------------------------------- the pipeline, offline
def run(chapter, cap=20_000, answers=1, label_live=False, r08=None):
    r08 = r08 or chapter("r08_live")
    data = r08.Data()
    config = live_config(cap) if label_live else bd.LiveConfig("offline-extract", "offline-query", "offline-titan-embed", "offline", "", cap, 256, True)
    budget = bd.TokenBudget(cap)
    bedrock = bd.Bedrock(bd.OfflineBedrock(256), config, budget)
    return r08, data, r08.run_pipeline(bedrock, data, answers), budget


def test_the_full_offline_run_completes_every_stage_inside_the_cap(chapter):
    _, _, res, budget = run(chapter)
    assert all(res["complete"].values()) and res["stopped"] is None and budget.used <= 20_000
    stages = res["budget"]["by_stage"]
    assert (stages["embed"]["calls"], stages["extract"]["calls"], stages["keywords"]["calls"], stages["answers"]["calls"]) == (57, 24, 15, 2)
    assert res["extraction"]["documents_done"] == 24 and res["keywords"] == {"valid": 15, "fallback": 0, "total": 15}
    assert set(res["retrieval"]) == set(chapter("r08_live").METHODS)
    assert all(a["valid"] for a in res["answers"]) and res["predictions"] == []


def test_every_cap_is_respected_and_nothing_crashes_when_it_bites(chapter):
    for cap in (300, 2_000, 4_500, 8_000, 14_000, 16_000, 20_000):
        _, _, res, budget = run(chapter, cap=cap)
        assert budget.used <= cap and res["budget"]["used"] <= cap, cap


def test_a_tiny_cap_stops_in_the_embedding_stage_and_reports_it(chapter):
    _, _, res, _ = run(chapter, cap=1_000)
    assert res["complete"]["embed"] is False and res["stopped"]["stage"] == "embed"
    assert "vector" not in res["retrieval"] and "bm25" in res["retrieval"] and "random" in res["retrieval"]
    assert not res["answers"]


def test_a_graph_that_could_not_be_finished_is_never_half_built(chapter):
    _, _, res, budget = run(chapter, cap=6_000)  # enough for the embeddings, far too little for the whole graph
    assert res["complete"]["embed"] and res["complete"]["extract"] is False and res["stopped"]["stage"] == "extract"
    assert res["extraction"]["documents_done"] == 0 and budget.by_stage().get("extract") is None
    assert "vector" in res["retrieval"] and "hybrid" not in res["retrieval"]


def squeeze_keywords(chapter, monkeypatch, worst=12_000):
    """Make the keyword stage's worst case larger than what remains after the graph is built, deterministically."""
    r08 = chapter("r08_live")
    original = r08.make_plan

    def patched(data, answers=1):
        plan = original(data, answers)
        for s in plan["stages"]:
            if s["stage"] == "keywords":
                s["worst"] = worst
        return plan

    monkeypatch.setattr(r08, "make_plan", patched)
    return r08  # the same module object must be used for the run, or the patch would not apply


def test_keywords_are_skipped_when_they_cannot_all_fit(chapter, monkeypatch):
    r08 = squeeze_keywords(chapter, monkeypatch)
    _, _, res, _ = run(chapter, r08=r08)
    assert res["complete"]["extract"] and res["complete"]["keywords"] is False and res["stopped"]["stage"] == "keywords"
    assert res["keywords"]["valid"] == 0 and "hybrid" not in res["retrieval"] and not res["answers"]
    assert "vector" in res["retrieval"] and res["budget"]["by_stage"].get("keywords") is None


def test_the_results_file_is_ascii_and_holds_no_prompts_or_credentials(chapter, tmp_path):
    r08, _, res, _ = run(chapter)
    path = tmp_path / "r.json"
    r08.save_results(res, path)
    text = path.read_text(encoding="utf-8")
    text.encode("ascii")
    for forbidden in ("---Role---", "---Instructions---", "AKIA", "aws_secret", "Bearer"):
        assert forbidden not in text
    assert json.loads(text)["mode"] == "offline"


def test_offline_results_are_labelled_as_not_real(chapter, capsys):
    r08, _, res, _ = run(chapter)
    r08.print_report(res)
    out = capsys.readouterr().out
    assert "NOT real results" in out and "only scored on a live run" in out


def test_a_live_labelled_run_scores_all_seven_predictions(chapter):
    r08, _, res, _ = run(chapter, label_live=True)
    preds = {p["id"]: p for p in res["predictions"]}
    assert list(preds) == ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    assert preds["L2"]["holds"] is True and preds["L4"]["holds"] is True and preds["L6"]["holds"] is True and preds["L7"]["holds"] is True


def test_predictions_say_not_checked_when_a_stage_did_not_finish(chapter, monkeypatch):
    r08 = squeeze_keywords(chapter, monkeypatch)
    _, _, res, _ = run(chapter, label_live=True, r08=r08)
    preds = {p["id"]: p for p in res["predictions"]}
    assert preds["L3"]["holds"] is None and preds["L4"]["holds"] is None and preds["L7"]["holds"] is None
    assert preds["L6"]["holds"] is True


def test_a_failing_model_call_stops_the_stage_cleanly(chapter):
    r08 = chapter("r08_live")
    exc = type("ClientError", (Exception,), {})("denied")
    exc.response = {"Error": {"Code": "AccessDeniedException"}}

    class Flaky(bd.OfflineBedrock):
        def converse(self, **request):
            raise exc

    config = live_config()
    bedrock = bd.Bedrock(Flaky(256), config, bd.TokenBudget(20_000))
    res = r08.run_pipeline(bedrock, r08.Data(), 1)
    assert res["complete"]["embed"] and res["complete"]["extract"] is False and "Access denied" in res["stopped"]["reason"]


# ---------------------------------------------------------------- the command line
def test_plan_and_frozen_commands_need_no_aws(chapter, capsys, monkeypatch):
    r08 = chapter("r08_live")
    monkeypatch.setattr(bd, "make_client", lambda c: pytest.fail("must not create a client"))
    assert r08.main(["--plan"]) == 0 and "Token plan" in capsys.readouterr().out
    assert r08.main(["--frozen"]) == 0 and "untouched" in capsys.readouterr().out


def test_live_without_yes_spends_nothing_and_says_how_to_proceed(chapter, capsys, monkeypatch):
    r08 = chapter("r08_live")
    for k, v in LIVE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(bd, "make_client", lambda c: pytest.fail("must not create a client"))
    assert r08.main(["--live"]) == 2
    out = capsys.readouterr().out
    assert "Nothing was spent" in out and "--yes" in out


def test_live_with_missing_settings_explains_what_to_set(chapter, capsys, monkeypatch):
    r08 = chapter("r08_live")
    for k in LIVE_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(bd, "load_env", lambda: None)
    assert r08.main(["--live", "--yes"]) == 2
    assert "RAKA_EXTRACT_MODEL" in capsys.readouterr().out


def test_the_smoke_test_is_tiny(chapter, capsys, monkeypatch):
    r08 = chapter("r08_live")
    for k, v in LIVE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(bd, "make_client", lambda c: bd.OfflineBedrock(256))
    assert r08.main(["--live", "--smoke"]) == 0
    out = capsys.readouterr().out
    used = int(out.split("tokens used: ")[1].split(" of")[0])
    assert used < 100 and "request shapes work" in out.lower()


def test_offline_run_saves_results_and_report_reads_them_back(chapter, capsys, monkeypatch, tmp_path):
    r08 = chapter("r08_live")
    monkeypatch.setattr(r08, "RESULTS_FILE", tmp_path / "out" / "r.json")
    assert r08.main(["--offline", "--answers", "0"]) == 0
    capsys.readouterr()
    assert (tmp_path / "out" / "r.json").is_file()
    assert r08.main(["--report", str(tmp_path / "out" / "r.json")]) == 0 and "OFFLINE" in capsys.readouterr().out
    assert r08.main(["--report", str(tmp_path / "nope.json")]) == 2


def test_bad_arguments_are_refused(chapter):
    r08 = chapter("r08_live")
    assert r08.main(["--bogus"]) == 2 and r08.main(["--answers", "9"]) == 2 and r08.main(["--answers"]) == 2
