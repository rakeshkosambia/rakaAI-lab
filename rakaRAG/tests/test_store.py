"""Tests for raka_store (the index), Graph.remove_chunk and the r06 chapter. Offline only."""
import json
import random

import pytest

import raka_common as rc
import raka_extract as ex
import raka_store
from raka_fake_model import FakeExtractor, FakeKeywordModel, fake_summarize
from raka_graph import KnowledgeGraph
from raka_store import Index, atomic_write

DOCS = rc.load_corpus()
BY_ID = {d.id: d for d in DOCS}
EXTRA = " Extra sentence about memory and tools in the agent loop."


def build(docs=DOCS, model=None):
    index = Index(model or FakeExtractor())
    for d in docs:
        index.add_document(d)
    return index


def gj(index):
    return json.dumps(index.graph.to_dict(), sort_keys=True)


def same_state(a, b):
    return gj(a) == gj(b) and a.chunks == b.chunks and a.docs == b.docs


def edited(doc):
    return rc.Doc(doc.id, doc.title, doc.text.rstrip() + EXTRA)


# ---------------------------------------------------------------- remove_chunk (graph level)
def E(chunk, entities, relations=()):
    return ex.Extraction(chunk, [ex.Entity(n, "other", d, chunk) for n, d in entities],
                         [ex.Relation(a, b, ("x",), d, chunk) for a, b, d in relations], complete=True)


def test_remove_chunk_deletes_sole_source_items_and_keeps_shared_ones():
    g = KnowledgeGraph()
    g.add_all([E("a#0", [("A", "da"), ("B", "db")], [("A", "B", "ab")]), E("b#0", [("B", "db2"), ("C", "dc")], [("B", "C", "bc")])])
    report = g.remove_chunk("b#0")
    assert report["nodes_deleted"] == ["c"] and report["nodes_rebuilt"] == ["b"]
    assert report["edges_deleted"] == ["b|c"] and set(g.nodes) == {"a", "b"} and list(g.edges) == [("a", "b")]
    assert g.nodes["b"].descriptions == ["db"]


def test_remove_chunk_clears_only_the_summaries_it_invalidates():
    g = KnowledgeGraph()
    g.add_all([E(f"a#{i}", [("Big", f"fragment {i}"), ("Other", "same")]) for i in range(5)])
    g.compact(fake_summarize, max_fragments=3)
    assert g.nodes["big"].summary is not None
    report = g.remove_chunk("a#0")
    assert g.nodes["big"].summary is None and report["summaries_cleared"] >= 1


def test_removing_an_unknown_chunk_changes_nothing():
    g = KnowledgeGraph()
    g.add_extraction(E("a#0", [("A", "da")]))
    before = json.dumps(g.to_dict(), sort_keys=True)
    report = g.remove_chunk("zzz#9")
    assert json.dumps(g.to_dict(), sort_keys=True) == before and not any(report[k] for k in report if k != "summaries_cleared")


def test_removing_every_chunk_leaves_an_empty_graph():
    g = KnowledgeGraph()
    g.add_all([E("a#0", [("A", "da"), ("B", "db")], [("A", "B", "ab")]), E("a#1", [("A", "da2")])])
    g.remove_chunk("a#0")
    g.remove_chunk("a#1")
    assert g.nodes == {} and g.edges == {}


# ---------------------------------------------------------------- adding
def test_a_full_build_is_consistent_and_costs_one_call_per_chunk():
    index = build()
    assert index.verify() == [] and index.model_calls == len(index.chunks) == 40
    assert len(index.graph.nodes) == 21 and len(index.graph.edges) == 29


def test_adding_the_same_document_again_is_a_free_no_op():
    index = build()
    before = gj(index)
    report = index.add_document(BY_ID["04_hooks"])
    assert report["status"] == "unchanged" and report["model_calls"] == 0 and gj(index) == before


def test_adding_a_changed_document_under_an_old_id_is_refused():
    index = build()
    with pytest.raises(ValueError):
        index.add_document(edited(BY_ID["04_hooks"]))
    assert index.verify() == []


# ---------------------------------------------------------------- deleting
@pytest.mark.parametrize("doc_id", [d.id for d in DOCS])
def test_deleting_a_document_equals_never_having_indexed_it(doc_id):
    index = build()
    calls = index.model_calls
    report = index.delete_document(doc_id)
    scratch = build([d for d in DOCS if d.id != doc_id])
    assert report["status"] == "deleted" and report["model_calls"] == 0 and index.model_calls == calls
    assert same_state(index, scratch) and index.verify() == []


def test_the_delete_report_counts_each_node_once_and_is_consistent_with_the_graph():
    index = build()
    nodes_before, edges_before = len(index.graph.nodes), len(index.graph.edges)
    r = index.delete_document("04_hooks")
    assert nodes_before - r["nodes_deleted"] == len(index.graph.nodes)
    assert edges_before - r["edges_deleted"] == len(index.graph.edges)
    assert r["nodes_rebuilt"] <= len(index.graph.nodes) and r["chunks_removed"] == 5


def test_deleting_an_unknown_document_is_reported_not_raised():
    index = build()
    before = gj(index)
    assert index.delete_document("nope") == {"doc": "nope", "status": "not_found"} and gj(index) == before


def test_deleting_everything_leaves_an_empty_valid_index():
    index = build()
    for d in DOCS:
        index.delete_document(d.id)
    assert index.graph.nodes == {} and index.graph.edges == {} and index.chunks == {} and index.docs == {}
    assert index.verify() == []


def test_deleted_documents_vanish_from_retrieval():
    index = build()
    q = "What does a pre-tool hook do?"
    assert any(c.doc_id == "04_hooks" for c in index.retriever(FakeKeywordModel()).retrieve(q, "hybrid").chunks)
    index.delete_document("04_hooks")
    ctx = index.retriever(FakeKeywordModel()).retrieve(q, "hybrid")
    assert ctx.missing_sources == 0 and all(c.doc_id != "04_hooks" for c in ctx.chunks)


def test_deleting_clears_summaries_on_touched_items_only():
    index = build()
    index.graph.compact(fake_summarize, max_fragments=3)
    before = {k: (n.summary, any(c.startswith("04_hooks#") for c in n.mentions)) for k, n in index.graph.nodes.items()}
    assert any(s for s, _ in before.values()) and any(touched for _, touched in before.values())
    report = index.delete_document("04_hooks")
    assert report["summaries_cleared"] > 0 and index.verify() == []
    for key, node in index.graph.nodes.items():
        summary, touched = before[key]
        assert node.summary is (None if touched else summary), key


# ---------------------------------------------------------------- the cache
def test_drop_cache_keeps_extractions_that_another_document_still_needs():
    twin = rc.Doc("twin", "t", BY_ID["04_hooks"].text)
    index = build([BY_ID["04_hooks"], twin])
    assert index.model_calls == 5  # the twin's chunks all came from the cache
    index.delete_document("04_hooks", drop_cache=True)
    assert index.verify() == [] and len(index.cache) == 5
    index.delete_document("twin", drop_cache=True)
    assert index.cache == {} and index.verify() == []


def test_without_drop_cache_a_deleted_documents_answers_are_kept_for_next_time():
    index = build()
    index.delete_document("04_hooks")
    calls = index.model_calls
    report = index.add_document(BY_ID["04_hooks"])
    assert report["model_calls"] == 0 and report["cache_hits"] == 5 and index.model_calls == calls


def test_a_changed_prompt_limit_does_not_reuse_old_answers():
    a, b = Index(FakeExtractor(), max_rows=12), Index(FakeExtractor(), max_rows=6)
    doc = BY_ID["04_hooks"]
    a.add_document(doc)
    b.cache = dict(a.cache)  # same texts, different row limit: the keys differ
    assert b.add_document(doc)["cache_hits"] == 0


# ---------------------------------------------------------------- updating
def test_an_update_re_extracts_only_the_chunks_whose_text_changed():
    index = build()
    result = index.update_document(edited(BY_ID["04_hooks"]))
    assert result["status"] == "updated" and result["model_calls"] < result["chunks"] and result["cache_hits"] >= 3
    docs = [edited(d) if d.id == "04_hooks" else d for d in DOCS]
    assert same_state(index, build(docs)) and index.verify() == []


def test_updating_with_identical_text_costs_nothing():
    index = build()
    before = gj(index)
    assert index.update_document(BY_ID["04_hooks"]) == {"doc": "04_hooks", "status": "unchanged", "model_calls": 0}
    assert gj(index) == before


def test_updating_an_unknown_document_adds_it():
    index = build([d for d in DOCS if d.id != "04_hooks"])
    assert index.update_document(BY_ID["04_hooks"])["status"] == "added"
    assert same_state(index, build())


# ---------------------------------------------------------------- failure leaves nothing behind
class Failing:
    def __init__(self, fail_on):
        self.inner, self.fail_on, self.calls = FakeExtractor(), fail_on, 0

    def __call__(self, prompt):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("down")
        return self.inner(prompt)


@pytest.mark.parametrize("nth", [1, 2, 3, 4])
def test_a_model_failure_part_way_through_a_document_rolls_everything_back(nth):
    model = Failing(fail_on=4 + nth)  # call numbers 5..8 land inside the second document
    index = build(DOCS[:1], model)
    snapshot = (gj(index), dict(index.chunks), dict(index.docs))
    with pytest.raises(RuntimeError):
        index.add_document(DOCS[1])
    assert (gj(index), index.chunks, index.docs) == snapshot and index.verify() == []
    model.fail_on = 0
    report = index.add_document(DOCS[1])
    assert report["status"] == "added" and report["cache_hits"] == nth - 1 and index.verify() == []
    assert same_state(index, build(DOCS[:2]))


# ---------------------------------------------------------------- random histories
@pytest.mark.parametrize("seed", range(6))
def test_any_history_of_adds_updates_and_deletes_equals_a_scratch_build_of_what_remains(seed):
    rng = random.Random(seed)
    index, current = Index(FakeExtractor()), {}
    for _ in range(50):
        doc = rng.choice(DOCS)
        if doc.id in current and rng.random() < 0.5:
            index.delete_document(doc.id, drop_cache=rng.random() < 0.3)
            del current[doc.id]
        elif doc.id in current:
            current[doc.id] = edited(doc) if current[doc.id].text == doc.text else doc
            index.update_document(current[doc.id])
        else:
            current[doc.id] = edited(doc) if rng.random() < 0.3 else doc
            index.add_document(current[doc.id])
        assert index.verify() == []
    assert same_state(index, build(list(current.values())))


# ---------------------------------------------------------------- the doctor check catches damage
def test_verify_is_clean_after_normal_use_and_flags_each_kind_of_damage():
    def damaged(change):
        index = build()
        change(index)
        return " | ".join(index.verify())

    assert damaged(lambda i: i.chunks.pop("04_hooks#0")) and "not stored" in damaged(lambda i: i.chunks.pop("04_hooks#0"))
    assert "belongs to no document" in damaged(lambda i: i.docs.pop("04_hooks"))
    assert "missing chunk" in damaged(lambda i: i.graph.nodes["hook"].mentions.__setitem__("ghost#0", {"type": "other", "description": "d", "surface": "Hook"}))
    assert "missing node" in damaged(lambda i: i.graph.nodes.pop("hook"))
    assert "differs from a fresh rebuild" in damaged(lambda i: i.graph.nodes["hook"].mentions["04_hooks#0"].__setitem__("description", "tampered"))
    assert "cache lacks" in damaged(lambda i: i.cache.clear())


# ---------------------------------------------------------------- saving and loading
def test_save_and_load_round_trip(tmp_path):
    index = build()
    index.save(tmp_path / "idx")
    back = Index.load(tmp_path / "idx", FakeExtractor())
    assert same_state(back, index) and back.cache == index.cache and back.verify() == []
    assert back.add_document(BY_ID["04_hooks"])["status"] == "unchanged"


def test_saved_files_are_ascii_and_do_not_depend_on_the_order_documents_were_added(tmp_path):
    build().save(tmp_path / "a")
    build(list(reversed(DOCS))).save(tmp_path / "b")
    for name in ("chunks.json", "cache.json", "docs.json", "graph.json", "meta.json"):
        text = (tmp_path / "a" / name).read_text(encoding="utf-8")
        text.encode("ascii")
        assert text == (tmp_path / "b" / name).read_text(encoding="utf-8"), name


def test_save_refuses_an_inconsistent_index(tmp_path):
    index = build()
    index.chunks.pop("04_hooks#0")
    with pytest.raises(ValueError):
        index.save(tmp_path / "idx")
    assert not (tmp_path / "idx").exists()


def test_load_refuses_a_damaged_or_incompatible_index(tmp_path):
    build().save(tmp_path / "ok")

    def copy_and_break(name, mutate):
        import shutil
        target = tmp_path / name
        shutil.copytree(tmp_path / "ok", target)
        mutate(target)
        return target

    cases = {
        "missing": lambda t: (t / "graph.json").unlink(),
        "malformed": lambda t: (t / "cache.json").write_text("{broken", encoding="utf-8"),
        "tampered": lambda t: (t / "docs.json").write_text(json.dumps({}), encoding="utf-8"),
        "version": lambda t: (t / "meta.json").write_text(json.dumps({"version": 99}), encoding="utf-8"),
        "prompt": lambda t: (t / "meta.json").write_text(
            json.dumps({**json.loads((t / "meta.json").read_text(encoding="utf-8")), "prompt_version": "old"}), encoding="utf-8"),
    }
    for name, mutate in cases.items():
        with pytest.raises(ValueError):
            Index.load(copy_and_break(name, mutate), FakeExtractor())


def test_a_failed_write_leaves_the_existing_file_untouched(tmp_path, monkeypatch):
    target = tmp_path / "f.json"
    atomic_write(target, "old")
    monkeypatch.setattr(raka_store.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        atomic_write(target, "new")
    assert target.read_text(encoding="utf-8") == "old" and not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------- the chapter
def test_r06_every_delete_matches_a_scratch_build(chapter):
    r06 = chapter("r06_incremental")
    for d in DOCS:
        report, same = r06.delete_matches_scratch(d.id)
        assert same and report["model_calls"] == 0


def test_r06_command_line(chapter, capsys, tmp_path):
    r06 = chapter("r06_incremental")
    assert r06.main(["--offline"]) == 0 and "= scratch" in capsys.readouterr().out
    assert r06.main(["--delete", "04_hooks"]) == 0 and "equals an index that never saw it: True" in capsys.readouterr().out
    assert r06.main(["--demo-update"]) == 0 and "graph equals a scratch build of the edited corpus: True" in capsys.readouterr().out
    assert r06.main(["--demo-fail"]) == 0 and "graph unchanged by the failed add: True" in capsys.readouterr().out
    assert r06.main(["--save", str(tmp_path / "x")]) == 0 and "verify: clean" in capsys.readouterr().out
    assert r06.main(["--delete", "nope"]) == 2
    assert r06.main(["--bogus"]) == 2
