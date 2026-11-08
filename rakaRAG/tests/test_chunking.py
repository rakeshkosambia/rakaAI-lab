"""Tests for raka_chunking and the r02 chapter. Offline only."""
import pytest

import raka_chunking as ck
import raka_common as rc

DOCS = rc.load_corpus()


def doc(text, doc_id="d"):
    return rc.Doc(doc_id, "t", text)


# ---------------------------------------------------------------- size rules
@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)])
def test_bad_sizes_are_refused(size, overlap):
    with pytest.raises(ValueError):
        ck.chunk_fixed(doc("one two three"), size, overlap)


def test_recursive_refuses_a_zero_size():
    with pytest.raises(ValueError):
        ck.chunk_recursive(doc("one two"), 0)


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError):
        ck.chunk_doc(doc("one two"), "magic")


# ---------------------------------------------------------------- shared properties
@pytest.mark.parametrize("strategy,size,overlap", [("doc", 0, 0), ("fixed", 40, 0), ("fixed", 20, 5), ("recursive", 40, 0), ("recursive", 20, 0), ("recursive", 5, 0)])
def test_every_chunk_is_the_exact_source_text_at_its_span(strategy, size, overlap):
    for d in DOCS:
        for c in ck.chunk_doc(d, strategy, size, overlap):
            assert d.text[c.start:c.end] == c.text
            assert c.text == c.text.strip() and c.text


@pytest.mark.parametrize("strategy,size,overlap", [("fixed", 40, 0), ("fixed", 20, 5), ("recursive", 40, 0), ("recursive", 20, 0), ("recursive", 5, 0)])
def test_no_chunk_is_longer_than_size_and_no_word_is_lost(strategy, size, overlap):
    for d in DOCS:
        chunks = ck.chunk_doc(d, strategy, size, overlap)
        assert all(c.words <= size for c in chunks)
        covered = set()
        for c in chunks:
            covered.update(range(c.start, c.end))
        wanted = {i for i, ch in enumerate(d.text) if not ch.isspace()}
        assert wanted <= covered


@pytest.mark.parametrize("strategy,size,overlap", [("fixed", 20, 5), ("recursive", 20, 0)])
def test_chunk_numbers_run_from_zero_and_ids_are_unique(strategy, size, overlap):
    chunks = ck.chunk_corpus(DOCS, strategy, size, overlap)
    assert len({c.id for c in chunks}) == len(chunks)
    for d in DOCS:
        mine = [c for c in chunks if c.doc_id == d.id]
        assert [c.index for c in mine] == list(range(len(mine)))


def test_empty_and_blank_text_make_no_chunks():
    for strategy in ("doc", "fixed", "recursive"):
        assert ck.chunk_doc(doc(""), strategy, 10, 0) == []
        assert ck.chunk_doc(doc("  \n\n  "), strategy, 10, 0) == []


# ---------------------------------------------------------------- fixed window
def test_fixed_window_overlaps_by_the_requested_number_of_words():
    d = doc(" ".join(f"w{i}" for i in range(25)))
    chunks = ck.chunk_fixed(d, 10, 3)
    assert [c.words for c in chunks] == [10, 10, 10, 4]
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]


def test_fixed_window_does_not_emit_a_tail_made_only_of_overlap():
    d = doc(" ".join(f"w{i}" for i in range(20)))
    chunks = ck.chunk_fixed(d, 10, 5)
    assert [c.words for c in chunks] == [10, 10, 10]  # windows at 0, 5, 10; the last one reaches the end
    assert chunks[-1].text.split()[-1] == "w19"


def test_fixed_window_with_text_shorter_than_size_is_one_chunk():
    assert len(ck.chunk_fixed(doc("just three words"), 10, 2)) == 1


# ---------------------------------------------------------------- recursive
def test_recursive_keeps_short_text_whole():
    assert len(ck.chunk_recursive(doc("A short note. Two sentences."), 40)) == 1


def test_recursive_splits_on_sentences_before_cutting_words():
    d = doc("First sentence has some words. Second sentence has other words. Third one ends it.")
    chunks = ck.chunk_recursive(d, 6)
    assert [c.text for c in chunks] == ["First sentence has some words.", "Second sentence has other words.", "Third one ends it."]


def test_recursive_merges_small_neighbours_up_to_the_size():
    d = doc("One two. Three four. Five six. Seven eight.")
    chunks = ck.chunk_recursive(d, 4)
    assert [c.text for c in chunks] == ["One two. Three four.", "Five six. Seven eight."]


def test_recursive_prefers_blank_lines_over_sentences():
    d = doc("Alpha beta gamma. Delta.\n\nEpsilon zeta eta. Theta.")
    chunks = ck.chunk_recursive(d, 4)
    assert [c.text for c in chunks] == ["Alpha beta gamma. Delta.", "Epsilon zeta eta. Theta."]


def test_recursive_falls_back_to_words_for_a_run_with_no_boundaries():
    d = doc(" ".join(["word"] * 25))
    chunks = ck.chunk_recursive(d, 10)
    assert all(c.words <= 10 for c in chunks) and sum(c.words for c in chunks) == 25


def test_recursive_cuts_fewer_sentences_in_half_than_a_fixed_window():
    def cut(chunks):  # chunks that do not end where a sentence ends
        return sum(1 for c in chunks if not c.text.rstrip().endswith((".", ":")))
    fixed = ck.chunk_corpus(DOCS, "fixed", 20, 0)
    recursive = ck.chunk_corpus(DOCS, "recursive", 20, 0)
    assert cut(recursive) < cut(fixed)


# ---------------------------------------------------------------- scoring and chapter
def test_evaluate_chunk_retrieval_counts_coverage_mrr_and_words():
    questions = [{"id": "q", "type": "global", "question": "x", "gold_docs": ["a", "b"], "expect": []}]
    chunks = [ck.Chunk("z", 0, "one two", 0, 7), ck.Chunk("a", 0, "three four five", 0, 15), ck.Chunk("a", 1, "six", 0, 3)]
    result = rc.evaluate_chunk_retrieval(lambda q: chunks, questions, k=3)
    s = result["summary"]["all"]
    assert s["coverage"] == 0.5 and s["mrr"] == 0.5 and s["words"] == 6


def test_r02_scores_every_configuration_and_stays_in_range(chapter):
    r02 = chapter("r02_chunking")
    results = r02.run_eval(3)
    assert [r["label"] for r in results] == [c[0] for c in r02.CONFIGS]
    for r in results:
        assert r["chunks"] >= 10 and r["avg_words"] > 0
        for name in ("bm25", "vector"):
            for group in ("all", "local", "global"):
                s = r[name]["summary"][group]
                assert 0.0 <= s["coverage"] <= 1.0 and 0.0 <= s["mrr"] <= 1.0


def test_r02_whole_document_row_matches_r01(chapter):
    r02, r01 = chapter("r02_chunking"), chapter("r01_baselines")
    whole = r02.run_eval(3)[0]
    base = r01.run_eval(3)
    assert whole["bm25"]["summary"]["all"]["coverage"] == base["bm25"]["summary"]["all"]["recall"]
    assert whole["vector"]["summary"]["all"]["coverage"] == base["vector"]["summary"]["all"]["recall"]


def test_r02_chunking_shrinks_the_prompt(chapter):
    r02 = chapter("r02_chunking")
    results = {r["label"]: r for r in r02.run_eval(3)}
    assert results["recursive 40"]["bm25"]["summary"]["all"]["words"] < results["whole document"]["bm25"]["summary"]["all"]["words"] / 2


def test_r02_command_line(chapter, capsys):
    r02 = chapter("r02_chunking")
    assert r02.main(["--offline"]) == 0
    assert "whole document" in capsys.readouterr().out
    assert r02.main(["--show", "04_hooks", "--strategy", "fixed", "--size", "30", "--overlap", "10"]) == 0
    assert "04_hooks#0" in capsys.readouterr().out
    assert r02.main(["--show", "nope"]) == 2
    assert r02.main(["--show", "04_hooks", "--strategy", "fixed", "--size", "10", "--overlap", "10"]) == 2
    assert r02.main(["--bogus"]) == 2
