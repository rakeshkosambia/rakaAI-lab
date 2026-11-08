"""Tests for raka_graph and the r04 chapter. Offline only."""
import json

import pytest

import raka_extract as ex
from raka_fake_model import fake_summarize
from raka_graph import KnowledgeGraph, canonical_key, chunk_sort_key


def E(chunk, entities=(), relations=()):
    """Build a validated-looking Extraction by hand: entities (name, type, desc), relations (a, b, kw, desc)."""
    return ex.Extraction(
        chunk,
        [ex.Entity(n, t, d, chunk) for n, t, d in entities],
        [ex.Relation(a, b, tuple(k.split(",")), d, chunk) for a, b, k, d in relations],
        complete=True,
    )


# ---------------------------------------------------------------- identity
@pytest.mark.parametrize("a,b", [("Hook", "hooks"), ("  agent   LOOP ", "Agent Loop"), ("The Tool", "tool"), ("Memories", "memory")])
def test_different_spellings_of_one_name_share_a_key(a, b):
    assert canonical_key(a) == canonical_key(b)


def test_different_things_do_not_share_a_key():
    assert canonical_key("Context Window") != canonical_key("Context Compaction")
    assert canonical_key("Tool") != canonical_key("Tools Gateway")


def test_an_empty_name_has_no_key():
    with pytest.raises(ValueError):
        canonical_key("   ")


def test_chunk_ids_sort_by_number_not_by_text():
    ids = ["d#10", "d#2", "a#1", "d#1"]
    assert sorted(ids, key=chunk_sort_key) == ["a#1", "d#1", "d#2", "d#10"]


# ---------------------------------------------------------------- merging nodes
def test_the_same_entity_in_two_chunks_is_one_node_with_both_sources():
    g = KnowledgeGraph()
    g.add_all([E("a#0", [("Hook", "mechanism", "Runs at a point.")]), E("b#0", [("hooks", "mechanism", "Adds checks.")])])
    assert len(g.nodes) == 1
    node = g.nodes["hook"]
    assert node.sources == ["a#0", "b#0"] and node.descriptions == ["Runs at a point.", "Adds checks."]


def test_the_type_is_the_one_most_mentions_agree_on():
    g = KnowledgeGraph()
    g.add_all([E("a#0", [("X", "concept", "d1")]), E("a#1", [("X", "mechanism", "d2")]), E("a#2", [("X", "mechanism", "d3")])])
    assert g.nodes["x"].type == "mechanism"


def test_a_tie_on_type_goes_to_the_earliest_chunk():
    g = KnowledgeGraph()
    g.add_all([E("a#1", [("X", "concept", "d2")]), E("a#0", [("X", "mechanism", "d1")])])  # added out of order
    assert g.nodes["x"].type == "mechanism"


def test_identical_descriptions_are_collected_once_and_order_follows_chunk_order():
    g = KnowledgeGraph()
    g.add_all([E("a#10", [("X", "other", "late")]), E("a#2", [("X", "other", "early")]), E("a#3", [("X", "other", "early")])])
    assert g.nodes["x"].descriptions == ["early", "late"]


def test_adding_the_same_chunk_twice_changes_nothing():
    g = KnowledgeGraph()
    ext = E("a#0", [("A", "other", "da"), ("B", "other", "db")], [("A", "B", "links", "A links B.")])
    g.add_extraction(ext)
    before = json.dumps(g.to_dict(), sort_keys=True)
    g.add_extraction(ext)
    assert json.dumps(g.to_dict(), sort_keys=True) == before
    assert g.edge_between("A", "B").weight == 1


def test_re_extracting_a_chunk_replaces_its_old_mention():
    g = KnowledgeGraph()
    g.add_extraction(E("a#0", [("A", "other", "old words")]))
    g.add_extraction(E("a#0", [("A", "other", "new words")]))
    assert g.nodes["a"].descriptions == ["new words"] and g.nodes["a"].sources == ["a#0"]


def test_the_surface_name_shown_is_the_most_common_spelling():
    g = KnowledgeGraph()
    g.add_all([E("a#0", [("Hooks", "other", "d1")]), E("a#1", [("Hook", "other", "d2")]), E("a#2", [("Hook", "other", "d3")])])
    assert g.nodes["hook"].name == "Hook"


def test_a_node_with_no_description_gets_a_plain_fallback():
    g = KnowledgeGraph()
    g.add_extraction(E("a#0", [("Thing", "other", "")]))
    assert g.nodes["thing"].descriptions == ["Entity Thing"]


# ---------------------------------------------------------------- aliases (identity is declared)
def test_a_declared_alias_merges_two_names():
    g = KnowledgeGraph({"MCP": "Model Context Protocol"})
    g.add_all([E("a#0", [("MCP", "protocol", "d1")]), E("b#0", [("Model Context Protocol", "protocol", "d2")])])
    assert list(g.nodes) == ["model context protocol"] and len(g.nodes["model context protocol"].sources) == 2


def test_without_a_declared_alias_the_names_stay_apart():
    g = KnowledgeGraph()
    g.add_all([E("a#0", [("MCP", "protocol", "d1")]), E("b#0", [("Model Context Protocol", "protocol", "d2")])])
    assert len(g.nodes) == 2


def test_alias_chains_and_late_declarations_are_refused():
    g = KnowledgeGraph({"b": "c"})
    with pytest.raises(ValueError):
        g.declare_alias("a", "b")  # b is itself an alias
    with pytest.raises(ValueError):
        KnowledgeGraph({"x": "y", "y": "z"})  # y would be both a target and an alias
    g.add_extraction(E("a#0", [("Q", "other", "d")]))
    with pytest.raises(ValueError):
        g.declare_alias("r", "q")  # earlier mentions would stay split


# ---------------------------------------------------------------- edges
def two_nodes(chunk, *relations):
    return E(chunk, [("A", "other", "da"), ("B", "other", "db")], relations)


def test_a_reversed_relation_is_the_same_edge_and_weight_counts_chunks():
    g = KnowledgeGraph()
    g.add_all([two_nodes("a#0", ("A", "B", "blocks", "A blocks B.")), two_nodes("b#0", ("B", "A", "limits,blocks", "B limited by A."))])
    assert len(g.edges) == 1
    edge = g.edge_between("a", "b")
    assert edge.weight == 2 and edge.keywords == ["blocks", "limits"] and edge.sources == ["a#0", "b#0"]


def test_a_relation_to_a_missing_node_is_refused():
    g = KnowledgeGraph()
    with pytest.raises(ValueError):
        g.add_extraction(E("a#0", [("A", "other", "d")], [("A", "Ghost", "x", "d")]))


def test_two_names_that_resolve_to_one_node_make_no_self_loop():
    g = KnowledgeGraph({"MCP": "Model Context Protocol"})
    g.add_extraction(E("a#0", [("MCP", "protocol", "d1"), ("Model Context Protocol", "protocol", "d2")], [("MCP", "Model Context Protocol", "same", "d")]))
    assert g.edges == {}


# ---------------------------------------------------------------- reading the graph
def chain():
    g = KnowledgeGraph()
    g.add_all([
        E("a#0", [("A", "other", "da"), ("B", "other", "db")], [("A", "B", "x", "ab")]),
        E("a#1", [("B", "other", "db"), ("C", "other", "dc")], [("B", "C", "x", "bc")]),
        E("a#2", [("D", "other", "dd")]),
    ])
    return g


def test_neighbours_degree_and_path():
    g = chain()
    assert g.neighbours("B") == ["a", "c"] and g.degree("b") == 2
    assert g.shortest_path("A", "C") == ["A", "B", "C"]
    assert g.shortest_path("A", "A") == ["A"]


def test_no_path_to_an_isolated_or_unknown_node():
    g = chain()
    assert g.shortest_path("A", "D") is None and g.shortest_path("A", "Nope") is None


def test_components_and_stats():
    g = chain()
    assert [len(c) for c in g.components()] == [3, 1]
    s = g.stats()
    assert (s["nodes"], s["edges"], s["components"], s["isolated"], s["largest_component"]) == (4, 2, 2, 1, 3)
    assert s["avg_degree"] == pytest.approx(1.0)


def test_an_empty_graph_has_sane_stats():
    s = KnowledgeGraph().stats()
    assert s["nodes"] == 0 and s["components"] == 0 and s["avg_degree"] == 0.0


# ---------------------------------------------------------------- summaries
def test_compact_summarises_only_long_piles_and_counts_the_calls():
    g = KnowledgeGraph()
    g.add_all([E(f"a#{i}", [("Big", "other", f"fragment number {i} with some words"), ("Small", "other", "just one")]) for i in range(5)])
    calls = []

    def summarize(name, descriptions):
        calls.append((name, len(descriptions)))
        return "short summary"

    assert g.compact(summarize, max_fragments=3) == 1
    assert calls == [("Big", 5)] and g.nodes["big"].description == "short summary"
    assert g.nodes["small"].summary is None
    assert g.compact(summarize, max_fragments=3) == 0  # already summarised: no second call


def test_a_new_mention_clears_a_stale_summary():
    g = KnowledgeGraph()
    g.add_all([E(f"a#{i}", [("Big", "other", f"fragment {i}")]) for i in range(5)])
    g.compact(fake_summarize, max_fragments=3)
    assert g.nodes["big"].summary is not None
    g.add_extraction(E("a#9", [("Big", "other", "a brand new fact")]))
    assert g.nodes["big"].summary is None


def test_fake_summarize_is_deterministic_and_short():
    parts = ["a long fragment about many things here", "tiny", "mid size one", "tiny"]
    assert fake_summarize("x", parts) == fake_summarize("x", parts) == "tiny mid size one"


# ---------------------------------------------------------------- persistence
def test_save_and_load_round_trip_exactly(tmp_path):
    g = chain()
    path = tmp_path / "g.json"
    g.save(path)
    back = KnowledgeGraph.load(path)
    assert back.to_dict() == g.to_dict()
    assert back.shortest_path("A", "C") == ["A", "B", "C"]


def test_the_saved_file_is_ascii_and_identical_whatever_the_build_order(tmp_path):
    parts = [E("a#0", [("A", "other", "da")]), E("b#0", [("B", "other", "caf" + chr(0xE9))]), E("c#0", [("C", "other", "dc")])]
    g1, g2 = KnowledgeGraph(), KnowledgeGraph()
    g1.add_all(parts)
    g2.add_all(list(reversed(parts)))
    g1.save(tmp_path / "1.json")
    g2.save(tmp_path / "2.json")
    text = (tmp_path / "1.json").read_text(encoding="utf-8")
    assert text == (tmp_path / "2.json").read_text(encoding="utf-8")
    text.encode("ascii")


@pytest.mark.parametrize("bad", [
    {"version": 99},
    [],
    {"version": 1, "nodes": [{"key": "a", "mentions": {}}]},
    {"version": 1, "nodes": [{"key": "a", "mentions": {"c#0": {"type": "t", "description": "d", "surface": "A"}}}],
     "edges": [{"a": "a", "b": "zz", "mentions": {}}]},
])
def test_a_malformed_graph_file_is_refused(bad):
    with pytest.raises(ValueError):
        KnowledgeGraph.from_dict(bad)


# ---------------------------------------------------------------- the chapter
def test_r04_builds_the_same_21_entities_whatever_the_chunking(chapter):
    r04 = chapter("r04_graph")
    nodes = {label: r04.build_graph(s, sz, ov).stats()["nodes"] for label, s, sz, ov in r04.CONFIGS}
    assert set(nodes.values()) == {21}


def test_r04_merge_shrinks_mentions_to_nodes_and_keeps_provenance(chapter):
    r04 = chapter("r04_graph")
    g = r04.build_graph()
    s = g.stats()
    assert s["mentions"] > 3 * s["nodes"]
    for node in g.nodes.values():
        assert node.sources and all("#" in c for c in node.sources)
    for edge in g.edges.values():
        assert edge.weight >= 1 and edge.a < edge.b and edge.a in g.nodes and edge.b in g.nodes


def test_r04_every_edge_is_backed_by_a_sentence_from_its_own_chunk(chapter):
    import raka_chunking as ck
    import raka_common as rc
    r04 = chapter("r04_graph")
    g = r04.build_graph()
    chunks = {c.id: c for c in ck.chunk_corpus(rc.load_corpus(), "recursive", 40, 0)}
    for edge in g.edges.values():
        for cid, m in edge.mentions.items():
            assert m["description"][:30].split()[0] in chunks[cid].text


def test_r04_command_line(chapter, capsys, tmp_path):
    r04 = chapter("r04_graph")
    assert r04.main(["--offline"]) == 0 and "whole document" in capsys.readouterr().out
    assert r04.main(["--top", "3"]) == 0 and "degree" in capsys.readouterr().out
    assert r04.main(["--node", "Subagent"]) == 0 and "linked to" in capsys.readouterr().out
    assert r04.main(["--path", "Hook", "Memory"]) == 0 and "hop" in capsys.readouterr().out
    assert r04.main(["--compact"]) == 0 and "Summary calls made" in capsys.readouterr().out
    target = tmp_path / "out" / "graph.json"
    assert r04.main(["--save", str(target)]) == 0 and target.is_file()
    assert r04.main(["--node", "Nope"]) == 2
    assert r04.main(["--path", "Hook", "Nope"]) == 2
    assert r04.main(["--bogus"]) == 2
