"""Tests for raka_eval, the r07 test set and the r07 chapter. Offline only."""
import json
import shutil

import pytest

import raka_chunking as ck
import raka_common as rc
import raka_eval as ev
from raka_fake_model import LEXICON, LEXICON_V2, FakeExtractor
from raka_store import Index

QUESTIONS = rc.load_questions(rc.QUESTIONS_V2_FILE)
DOCS = rc.load_corpus_v2()


# ---------------------------------------------------------------- the bootstrap
def test_identical_scores_give_a_zero_difference_with_a_zero_width_interval():
    assert ev.paired_bootstrap([0.5, 1.0, 0.0], [0.5, 1.0, 0.0]) == (0.0, 0.0, 0.0)


def test_a_consistent_advantage_has_an_interval_that_excludes_zero():
    mean, low, high = ev.paired_bootstrap([1.0] * 20, [0.5] * 20)
    assert mean == pytest.approx(0.5) and low > 0 and ev.clear_difference(low, high) == "higher"
    assert ev.clear_difference(*ev.paired_bootstrap([0.0] * 20, [0.5] * 20)[1:]) == "lower"


def test_a_noisy_difference_is_called_unclear():
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    b = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    mean, low, high = ev.paired_bootstrap(a, b)
    assert low < 0 < high and ev.clear_difference(low, high) == "unclear"


def test_the_bootstrap_repeats_exactly_and_the_seed_matters():
    a = [0.2, 0.9, 0.4, 0.7, 0.1, 0.55, 0.3, 0.85, 0.6, 0.15, 0.45, 0.95, 0.25, 0.65]
    b = [0.3, 0.5, 0.5, 0.2, 0.0, 0.35, 0.4, 0.45, 0.1, 0.25, 0.05, 0.55, 0.35, 0.15]
    assert ev.paired_bootstrap(a, b) == ev.paired_bootstrap(a, b)
    runs = [ev.paired_bootstrap(a, b, seed=s) for s in range(1, 8)]
    assert len({r[0] for r in runs}) == 1                 # the mean does not depend on the seed
    assert len({r[1:] for r in runs}) > 1                 # the interval does (two seeds can coincide on the discrete grid)
    assert max(r[1] for r in runs) - min(r[1] for r in runs) < 0.05 and max(r[2] for r in runs) - min(r[2] for r in runs) < 0.05  # but only slightly


@pytest.mark.parametrize("a,b", [([], []), ([1.0], [1.0, 2.0])])
def test_the_bootstrap_refuses_bad_input(a, b):
    with pytest.raises(ValueError):
        ev.paired_bootstrap(a, b)


# ---------------------------------------------------------------- lexical overlap
def test_overlap_is_a_share_between_zero_and_one():
    assert ev.lexical_overlap("What does a hook do?", ["A hook runs code."]) == 1.0
    assert ev.lexical_overlap("sandbox network", ["a hook"]) == 0.0
    assert ev.lexical_overlap("the of and", ["anything"]) == 0.0
    assert 0 < ev.lexical_overlap("hook and sandbox", ["a hook runs"]) < 1


def test_paraphrase_questions_really_share_fewer_words_with_their_answers_than_keyword_ones():
    by_id = {d.id: d for d in DOCS}

    def mean(style):
        values = [ev.lexical_overlap(q["question"], [by_id[g].text for g in q["gold_docs"]]) for q in QUESTIONS if q["style"] == style]
        return sum(values) / len(values)

    assert mean("keyword") - mean("paraphrase") >= 0.15


# ---------------------------------------------------------------- the frozen test set
def test_the_test_set_matches_the_hashes_recorded_before_the_first_run():
    assert ev.check_frozen() == []


def test_a_changed_question_or_document_is_detected(tmp_path):
    recorded = json.loads((rc.ROOT / "r07_evaluation" / "FROZEN.json").read_text(encoding="utf-8"))
    for key in ("questions_v2_sha256", "corpus_large_sha256"):
        fake = tmp_path / f"{key}.json"
        fake.write_text(json.dumps({**recorded, key: "0" * 64}), encoding="utf-8")
        problems = ev.check_frozen(fake)
        assert len(problems) == 1 and key in problems[0]


def test_hashing_ignores_line_ending_differences(tmp_path, monkeypatch):
    folder = tmp_path / "corpus_large"
    shutil.copytree(rc.CORPUS_LARGE_DIR, folder)
    for path in folder.glob("*.md"):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    monkeypatch.setattr(rc, "CORPUS_LARGE_DIR", folder)
    assert ev.frozen_hashes()["corpus_large_sha256"] == json.loads((rc.ROOT / "r07_evaluation" / "FROZEN.json").read_text(encoding="utf-8"))["corpus_large_sha256"]


# ---------------------------------------------------------------- the new documents and questions
def test_the_corpus_has_24_documents_with_unique_ids_and_sensible_lengths():
    assert len(DOCS) == 24 and len({d.id for d in DOCS}) == 24
    large = rc.load_corpus(rc.CORPUS_LARGE_DIR)
    assert [d.id[:2] for d in large] == [f"{n:02d}" for n in range(11, 25)]
    assert all(55 <= len(d.text.split()) <= 140 for d in large)


def test_the_question_set_is_valid_and_has_the_planned_shape():
    assert rc.validate_questions(QUESTIONS, DOCS) == []
    assert len(QUESTIONS) == 27
    assert sum(q["style"] == "keyword" for q in QUESTIONS) == 9 and sum(q["style"] == "paraphrase" for q in QUESTIONS) == 18
    assert sum(q["type"] == "local" for q in QUESTIONS) == 13 and sum(q["type"] == "global" for q in QUESTIONS) == 14
    assert not {q["id"] for q in QUESTIONS} & {q["id"] for q in rc.load_questions()}


def test_every_new_document_is_the_answer_to_at_least_one_question():
    gold = {g for q in QUESTIONS for g in q["gold_docs"]}
    assert {d.id for d in rc.load_corpus(rc.CORPUS_LARGE_DIR)} <= gold


def test_an_unknown_style_is_refused():
    bad = [dict(QUESTIONS[0], style="weird")]
    assert any("style must be" in p for p in rc.validate_questions(bad, DOCS))


def test_the_extended_lexicon_only_adds_terms_and_the_default_one_is_unchanged():
    assert LEXICON_V2[: len(LEXICON)] == LEXICON and len(LEXICON_V2) > len(LEXICON)
    assert FakeExtractor().lexicon is LEXICON and FakeExtractor(lexicon=LEXICON_V2).lexicon is LEXICON_V2


# ---------------------------------------------------------------- the random baseline
def test_random_ranking_is_a_repeatable_permutation_that_differs_by_question():
    chunks = ck.chunk_corpus(DOCS, "recursive", 40, 0)
    r = ev.RandomRetriever(chunks)
    a, b = r.ranking("question one"), r.ranking("question two")
    assert a == r.ranking("question one") and sorted(c.id for c in a) == sorted(c.id for c in chunks)
    assert a != b and ev.RandomRetriever(chunks, seed=1).ranking("question one") != a


# ---------------------------------------------------------------- cost
def test_tokens_are_estimated_from_words():
    assert ev.estimate_tokens(100) == 130 and ev.estimate_tokens(0) == 0


def test_indexing_cost_counts_one_call_per_distinct_chunk_text():
    twin = rc.Doc("twin", "t", DOCS[0].text)
    index = Index(FakeExtractor(lexicon=LEXICON_V2))
    index.add_document(DOCS[0])
    index.add_document(twin)
    cost = ev.indexing_cost(index)
    assert cost["calls"] == index.model_calls and cost["chunks"] == 2 * cost["calls"] and cost["prompt_words"] > cost["output_words"] > 0


# ---------------------------------------------------------------- the chapter
@pytest.fixture(scope="module")
def run():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("chapter_r07_evaluation", rc.ROOT / "r07_evaluation" / "code.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    prep = module.Prepared()
    return module, prep, module.evaluate_all(prep)


def test_r07_builds_the_expected_index(run):
    _, prep, _ = run
    assert len(prep.docs) == 24 and len(prep.chunks) == 90 and len(prep.questions) == 39
    assert prep.index.verify() == [] and prep.index.model_calls == 90


def test_r07_every_method_is_scored_on_every_question_in_range(run):
    r07, prep, results = run
    assert list(results) == list(r07.METHODS)
    for name, by_k in results.items():
        for k, r in by_k.items():
            assert len(r["rows"]) == 39 and all(0.0 <= v <= 1.0 for v in r["coverage"].values()), (name, k)


def test_r07_random_is_the_floor_and_bm25_clearly_beats_it(run):
    _, _, results = run
    assert results["random"][3]["coverage"]["all"] < 0.15
    assert all(results["bm25"][k]["coverage"]["all"] > 5 * results["random"][k]["coverage"]["all"] for k in (3, 6))


def test_r07_no_graph_mode_is_ever_clearly_better_than_bm25(run):
    r07, prep, results = run
    for group in ("paraphrase", "new", "all"):
        for k in (3, 6):
            for name, d in r07.differences(prep, results, k, group).items():
                if name != "vector":
                    assert d["verdict"] != "higher", (name, group, k)


def test_r07_mix_with_rerank_equals_bm25_in_coverage_by_construction(run):
    _, prep, results = run
    # the mix pool contains BM25's top 6 and the reranker is BM25, so coverage cannot differ at k <= 6
    for k in (3, 6):
        assert [r["coverage"] for r in results["mix+rerank"][k]["rows"]] == [r["coverage"] for r in results["bm25"][k]["rows"]]


def test_r07_graph_modes_cost_about_nine_times_the_words_and_one_call(run):
    _, _, results = run
    for m in ("local", "hybrid", "mix"):
        assert 8 <= results[m][3]["words"] / results["bm25"][3]["words"] <= 11 and results[m][3]["calls"] == 1
    assert results["bm25"][3]["calls"] == 0


def test_r07_the_recorded_outcome_of_each_prediction(run):
    r07, prep, results = run
    outcome = {p["id"]: p["holds"] for p in r07.check_predictions(prep, results)}
    assert len(outcome) == 8
    # recorded 2026-11-08: P2, P5 and P8 were wrong; the others held. If this changes, the results changed: update the README too.
    assert {k for k, v in outcome.items() if v} == {"P1", "P3", "P4", "P6", "P7"}


def test_r07_the_paraphrase_questions_are_less_hard_than_planned(run):
    # the honest finding behind the wrong P2: plain keyword search still scores well on them
    _, _, results = run
    assert results["bm25"][6]["coverage"]["paraphrase"] > 0.75 and results["bm25"][6]["coverage"]["keyword"] >= 0.95


def test_r07_command_line(chapter, capsys):
    r07 = chapter("r07_evaluation")
    assert r07.main(["--offline"]) == 0
    out = capsys.readouterr().out
    assert "Predictions from PREDICTIONS.md" in out and "untouched since it was frozen" in out
    assert r07.main(["--misses", "bm25"]) == 0 and "do not reach every gold document" in capsys.readouterr().out
    assert r07.main(["--frozen"]) == 0 and "untouched" in capsys.readouterr().out
    assert r07.main(["--misses", "nope"]) == 2
    assert r07.main(["--bogus"]) == 2
