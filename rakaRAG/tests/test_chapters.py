"""Tests for the chapters. Offline only: no model, no network."""
import pytest

import raka_common as rc


def test_r00_reports_the_test_set_and_passes(chapter, capsys):
    r00 = chapter("r00_setup")
    facts, problems = r00.describe()
    assert problems == []
    assert facts == {"documents": 10, "indexed_words": facts["indexed_words"], "questions": 12, "local": 8, "global": 4}
    assert r00.main([]) == 0
    assert "OK" in capsys.readouterr().out


def test_r01_bm25_puts_the_obvious_document_first(chapter):
    r01 = chapter("r01_baselines")
    index = r01.BM25Index(rc.load_corpus())
    assert index.search("What does a pre-tool hook do?", 1)[0][0] == "04_hooks"
    assert index.search("Which rule wins when allow and deny rules conflict?", 1)[0][0] == "03_permissions"


def test_r01_bm25_ignores_documents_with_no_matching_word(chapter):
    r01 = chapter("r01_baselines")
    index = r01.BM25Index(rc.load_corpus())
    assert index.search("zzzxqv unknownterm", 3) == []


def test_r01_rare_words_outweigh_common_ones(chapter):
    r01 = chapter("r01_baselines")
    index = r01.BM25Index(rc.load_corpus())
    assert index.idf["bm25"] > index.idf["tool"]


def test_r01_vector_search_is_repeatable(chapter):
    r01 = chapter("r01_baselines")
    docs = rc.load_corpus()
    one = r01.VectorIndex(docs).search("how are skills loaded on demand", 3)
    two = r01.VectorIndex(docs).search("how are skills loaded on demand", 3)
    assert one == two
    assert one[0][0] == "08_skills"


def test_r01_full_ranking_lists_every_document_once(chapter):
    r01 = chapter("r01_baselines")
    docs = rc.load_corpus()
    ranking = r01.full_ranking(r01.BM25Index(docs), docs)("pre-tool hook")
    assert sorted(ranking) == sorted(d.id for d in docs)
    assert ranking[0] == "04_hooks"


def test_r01_scores_are_in_range_and_every_question_is_scored(chapter):
    r01 = chapter("r01_baselines")
    results = r01.run_eval(3)
    for name in ("bm25", "vector"):
        assert len(results[name]["rows"]) == 12
        for group in ("all", "local", "global"):
            s = results[name]["summary"][group]
            assert 0.0 <= s["recall"] <= 1.0
            assert 0.0 <= s["mrr"] <= 1.0
    assert results["bm25"]["summary"]["local"]["n"] == 8


def test_r01_refuses_to_score_against_a_broken_question_file(chapter, monkeypatch, tmp_path):
    r01 = chapter("r01_baselines")
    broken = tmp_path / "q.json"
    broken.write_text('[{"id": "x", "type": "local", "question": "q", "gold_docs": ["missing"], "expect": []}]', encoding="utf-8")
    monkeypatch.setattr(rc, "QUESTIONS_FILE", broken)
    with pytest.raises(ValueError):
        r01.run_eval(3)


def test_r01_command_line_modes(chapter, capsys):
    r01 = chapter("r01_baselines")
    assert r01.main(["--offline"]) == 0
    out = capsys.readouterr().out
    assert "Summary" in out and "bm25" in out and "vector" in out
    assert r01.main(["--query", "What does a pre-tool hook do?"]) == 0
    assert "04_hooks" in capsys.readouterr().out
    assert r01.main(["--bogus"]) == 2
