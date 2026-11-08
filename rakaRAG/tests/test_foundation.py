"""Tests for raka_common: tokenising, the fake embedder, scoring and question validation."""
import math

import raka_common as rc


def test_tokenize_lowercases_drops_stop_words_and_stems():
    assert rc.tokenize("The Tools and the Policies") == ["tool", "policy"]


def test_tokenize_keeps_double_s_words_whole():
    assert rc.tokenize("access") == ["access"]


def test_tokenize_of_empty_text_is_empty():
    assert rc.tokenize("") == []
    assert rc.tokenize("the of and") == []


def test_hash_embed_is_unit_length_and_repeatable():
    v = rc.hash_embed("agent loop calls a tool")
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)
    assert v == rc.hash_embed("agent loop calls a tool")


def test_hash_embed_of_no_words_is_the_zero_vector():
    assert not any(rc.hash_embed("the of and"))


def test_cosine_of_a_text_with_itself_is_one_and_unrelated_is_lower():
    a = rc.hash_embed("permission rules allow deny ask")
    assert math.isclose(rc.cosine(a, a), 1.0, rel_tol=1e-9)
    assert rc.cosine(a, rc.hash_embed("embedding vector chunk citation")) < 0.5


def test_recall_and_reciprocal_rank():
    ranked = ["x", "a", "y", "b"]
    assert rc.recall_at_k(ranked, ["a", "b"], 2) == 0.5
    assert rc.recall_at_k(ranked, ["a", "b"], 4) == 1.0
    assert rc.recall_at_k(ranked, [], 3) == 0.0
    assert rc.reciprocal_rank(ranked, ["b", "a"]) == 0.5
    assert rc.reciprocal_rank(ranked, ["zzz"]) == 0.0


def test_evaluate_groups_local_and_global():
    questions = [
        {"id": "l", "type": "local", "question": "q1", "gold_docs": ["a"], "expect": []},
        {"id": "g", "type": "global", "question": "q2", "gold_docs": ["a", "b"], "expect": []},
    ]
    result = rc.evaluate(lambda q: ["a", "c", "b"], questions, k=2)
    assert result["summary"]["local"]["recall"] == 1.0
    assert result["summary"]["global"]["recall"] == 0.5
    assert result["summary"]["all"]["n"] == 2


def test_corpus_loads_ten_documents_with_unique_ids_and_titles():
    docs = rc.load_corpus()
    assert len(docs) == 10
    assert len({d.id for d in docs}) == 10
    assert all(d.title and d.text.strip() for d in docs)


def test_shipped_questions_validate_against_the_corpus():
    assert rc.validate_questions(rc.load_questions(), rc.load_corpus()) == []


def test_validate_questions_reports_each_kind_of_mistake():
    docs = rc.load_corpus()
    bad = [
        {"id": "a", "type": "local", "question": "q", "gold_docs": ["nope"], "expect": []},
        {"id": "a", "type": "weird", "question": "q", "gold_docs": [], "expect": []},
        {"id": "g", "type": "global", "question": "q", "gold_docs": ["04_hooks"], "expect": []},
        {"id": "m", "type": "local"},
    ]
    text = " | ".join(rc.validate_questions(bad, docs))
    for needle in ("not in the corpus", "duplicate id", "type must be", "gold_docs is empty", "at least 2 gold", "missing field"):
        assert needle in text, needle
