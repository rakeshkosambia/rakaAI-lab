"""A small knowledge graph for rakaRAG: nodes (entities), undirected edges (relations), merge rules.

Design (original code, ideas studied from LightRAG's merge step):
  * One node per entity, found by a normalised key (case, spacing, plural). Different names for the
    same thing are merged only through an alias table that WE declare; we never guess identity.
  * Every node and edge stores one "mention" per chunk it came from. All derived facts (type,
    descriptions, sources, weight) are computed from those mentions, so adding the same chunk twice
    changes nothing and removing a chunk later (r06) is a clean recompute.
  * The type is the one most mentions agree on. Descriptions are collected with exact duplicates
    dropped. When a node collects too many, compact() replaces them with one summary from an
    injected summarizer (a model call, so it is counted).
  * Edges are undirected: A-B and B-A are the same edge. weight = number of chunks that stated it.
Pure standard library.
"""
from __future__ import annotations

import json
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import raka_common as rc

SCHEMA_VERSION = 1
DESC_SEPARATOR = " | "


def canonical_key(name: str) -> str:
    """Normalised identity key: lower case, single spaces, no leading 'the', last word un-pluralised."""
    text = re.sub(r"\s+", " ", name.strip()).casefold()
    text = re.sub(r"^the ", "", text)
    if not text:
        raise ValueError("an entity name cannot be empty")
    head, _, last = text.rpartition(" ")
    return (head + " " if head else "") + rc._stem(last)


def chunk_sort_key(chunk_id: str) -> tuple[str, int]:
    """Order chunk ids as 'doc#2' before 'doc#10' (a plain string sort would not)."""
    doc, _, index = chunk_id.rpartition("#")
    return (doc, int(index)) if index.isdigit() else (chunk_id, -1)


def _ordered(mentions: dict) -> list[str]:
    return sorted(mentions, key=chunk_sort_key)


@dataclass
class Node:
    key: str
    mentions: dict[str, dict] = field(default_factory=dict)  # chunk_id -> {type, description, surface}
    summary: str | None = None

    @property
    def sources(self) -> list[str]:
        return _ordered(self.mentions)

    @property
    def type(self) -> str:
        votes = Counter(m["type"] for m in self.mentions.values())
        best = max(votes.values())
        return next(m["type"] for cid in self.sources for m in [self.mentions[cid]] if votes[m["type"]] == best)

    @property
    def name(self) -> str:
        votes = Counter(m["surface"] for m in self.mentions.values())
        best = max(votes.values())
        return next(m["surface"] for cid in self.sources for m in [self.mentions[cid]] if votes[m["surface"]] == best)

    @property
    def descriptions(self) -> list[str]:
        seen: list[str] = []
        for cid in self.sources:
            d = self.mentions[cid]["description"]
            if d and d not in seen:
                seen.append(d)
        return seen or [f"Entity {self.name}"]

    @property
    def description(self) -> str:
        return self.summary if self.summary is not None else DESC_SEPARATOR.join(self.descriptions)


@dataclass
class Edge:
    a: str  # node keys, a < b so the edge has one canonical form
    b: str
    mentions: dict[str, dict] = field(default_factory=dict)  # chunk_id -> {keywords, description}
    summary: str | None = None

    @property
    def sources(self) -> list[str]:
        return _ordered(self.mentions)

    @property
    def weight(self) -> int:
        return len(self.mentions)

    @property
    def keywords(self) -> list[str]:
        out: list[str] = []
        for cid in self.sources:
            for k in self.mentions[cid]["keywords"]:
                if k.casefold() not in (x.casefold() for x in out):
                    out.append(k)
        return out

    @property
    def descriptions(self) -> list[str]:
        seen: list[str] = []
        for cid in self.sources:
            d = self.mentions[cid]["description"]
            if d and d not in seen:
                seen.append(d)
        return seen

    @property
    def description(self) -> str:
        return self.summary if self.summary is not None else DESC_SEPARATOR.join(self.descriptions)


class KnowledgeGraph:
    def __init__(self, aliases: dict[str, str] | None = None):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str], Edge] = {}
        self.aliases: dict[str, str] = {}
        for alias, target in (aliases or {}).items():
            self.declare_alias(alias, target)

    # ---------------------------------------------------------------- identity
    def declare_alias(self, alias: str, target: str) -> None:
        """Say that `alias` means `target`. One hop only; identity is declared, never inferred."""
        a, t = canonical_key(alias), canonical_key(target)
        if a == t:
            return
        if t in self.aliases:
            raise ValueError(f"alias target {target!r} is itself an alias; point at the final name")
        if a in self.aliases.values():
            raise ValueError(f"{alias!r} is already the target of another alias; chains are not allowed")
        if self.nodes:
            raise ValueError("declare aliases before adding extractions, or earlier mentions would stay split")
        self.aliases[a] = t

    def resolve(self, name: str) -> str:
        key = canonical_key(name)
        return self.aliases.get(key, key)

    # ---------------------------------------------------------------- building
    def add_extraction(self, extraction) -> None:
        """Merge one chunk's validated extraction (raka_extract.Extraction) into the graph."""
        cid = extraction.chunk_id
        for e in extraction.entities:
            node = self.nodes.setdefault(self.resolve(e.name), Node(self.resolve(e.name)))
            node.mentions[cid] = {"type": e.type, "description": e.description, "surface": e.name}
            node.summary = None  # the old summary no longer covers every mention
        for r in extraction.relations:
            ka, kb = self.resolve(r.source), self.resolve(r.target)
            if ka not in self.nodes or kb not in self.nodes:
                raise ValueError(f"relation {r.source!r} - {r.target!r} names an entity that is not in the graph")
            if ka == kb:
                continue  # two names that resolve to one node make a self-loop; drop it
            edge_key = (ka, kb) if ka < kb else (kb, ka)
            edge = self.edges.setdefault(edge_key, Edge(*edge_key))
            edge.mentions[cid] = {"keywords": list(r.keywords), "description": r.description}
            edge.summary = None

    def add_all(self, extractions) -> None:
        for extraction in extractions:
            self.add_extraction(extraction)

    def remove_chunk(self, chunk_id: str) -> dict:
        """Forget everything one chunk contributed. A node or edge with no mentions left is deleted; one that
        still has other chunks behind it is kept (its type, descriptions and weight recompute on their own).
        A kept item loses its summary, because the summary no longer matches its mentions.

        Returns {nodes_deleted, nodes_rebuilt, edges_deleted, edges_rebuilt, edges_orphaned, summaries_cleared}.
        """
        report = {"nodes_deleted": [], "nodes_rebuilt": [], "edges_deleted": [], "edges_rebuilt": [],
                  "edges_orphaned": [], "summaries_cleared": 0}
        for key in sorted(self.nodes):
            node = self.nodes[key]
            if chunk_id not in node.mentions:
                continue
            del node.mentions[chunk_id]
            if not node.mentions:
                del self.nodes[key]
                report["nodes_deleted"].append(key)
            else:
                report["summaries_cleared"] += node.summary is not None
                node.summary = None
                report["nodes_rebuilt"].append(key)
        for ek in sorted(self.edges):
            edge = self.edges[ek]
            if chunk_id not in edge.mentions:
                continue
            del edge.mentions[chunk_id]
            if not edge.mentions:
                del self.edges[ek]
                report["edges_deleted"].append("|".join(ek))
            else:
                report["summaries_cleared"] += edge.summary is not None
                edge.summary = None
                report["edges_rebuilt"].append("|".join(ek))
        for ek in sorted(self.edges):  # defensive: an edge must never point at a node that no longer exists
            if ek[0] not in self.nodes or ek[1] not in self.nodes:
                del self.edges[ek]
                report["edges_orphaned"].append("|".join(ek))
        return report

    # ---------------------------------------------------------------- reading
    def neighbours(self, name: str) -> list[str]:
        key = self.resolve(name)
        out = [b if a == key else a for (a, b) in self.edges if key in (a, b)]
        return sorted(out)

    def degree(self, name: str) -> int:
        return len(self.neighbours(name))

    def edge_between(self, a: str, b: str) -> Edge | None:
        ka, kb = self.resolve(a), self.resolve(b)
        return self.edges.get((ka, kb) if ka < kb else (kb, ka))

    def shortest_path(self, a: str, b: str) -> list[str] | None:
        """Fewest-hops path between two entities as display names, or None. Ties break alphabetically."""
        start, goal = self.resolve(a), self.resolve(b)
        if start not in self.nodes or goal not in self.nodes:
            return None
        previous = {start: None}
        queue = deque([start])
        while queue:
            here = queue.popleft()
            if here == goal:
                break
            for nxt in self.neighbours(here):
                if nxt not in previous:
                    previous[nxt] = here
                    queue.append(nxt)
        if goal not in previous:
            return None
        path, cur = [], goal
        while cur is not None:
            path.append(self.nodes[cur].name)
            cur = previous[cur]
        return path[::-1]

    def components(self) -> list[list[str]]:
        """Connected groups of node keys, largest first."""
        seen: set[str] = set()
        groups = []
        for start in sorted(self.nodes):
            if start in seen:
                continue
            group, queue = [], deque([start])
            seen.add(start)
            while queue:
                here = queue.popleft()
                group.append(here)
                for nxt in self.neighbours(here):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            groups.append(sorted(group))
        return sorted(groups, key=lambda g: (-len(g), g[0]))

    def stats(self) -> dict:
        degrees = {k: self.degree(k) for k in self.nodes}
        comps = self.components()
        mentions = sum(len(n.mentions) for n in self.nodes.values())
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "mentions": mentions,
            "components": len(comps),
            "largest_component": len(comps[0]) if comps else 0,
            "isolated": sum(1 for d in degrees.values() if d == 0),
            "avg_degree": (2 * len(self.edges) / len(self.nodes)) if self.nodes else 0.0,
            "top": sorted(((self.nodes[k].name, d) for k, d in degrees.items()), key=lambda x: (-x[1], x[0]))[:5],
        }

    # ---------------------------------------------------------------- summaries (model calls)
    def compact(self, summarize, max_fragments: int = 3) -> int:
        """Replace a long pile of descriptions with one summary. Returns how many summarizer calls were made."""
        calls = 0
        for key in sorted(self.nodes):
            node = self.nodes[key]
            if node.summary is None and len(node.descriptions) > max_fragments:
                node.summary = summarize(node.name, node.descriptions)
                calls += 1
        for ek in sorted(self.edges):
            edge = self.edges[ek]
            if edge.summary is None and len(edge.descriptions) > max_fragments:
                edge.summary = summarize(f"{self.nodes[ek[0]].name} - {self.nodes[ek[1]].name}", edge.descriptions)
                calls += 1
        return calls

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "aliases": dict(sorted(self.aliases.items())),
            "nodes": [
                {"key": k, "summary": n.summary, "mentions": {c: n.mentions[c] for c in n.sources}}
                for k, n in sorted(self.nodes.items())
            ],
            "edges": [
                {"a": a, "b": b, "summary": e.summary, "mentions": {c: e.mentions[c] for c in e.sources}}
                for (a, b), e in sorted(self.edges.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        if not isinstance(data, dict) or data.get("version") != SCHEMA_VERSION:
            raise ValueError(f"not a version {SCHEMA_VERSION} graph file")
        graph = cls()
        graph.aliases = dict(data.get("aliases", {}))
        for n in data.get("nodes", []):
            node = Node(n["key"], dict(n["mentions"]), n.get("summary"))
            if not node.mentions:
                raise ValueError(f"node {n['key']!r} has no mentions")
            graph.nodes[node.key] = node
        for e in data.get("edges", []):
            if e["a"] not in graph.nodes or e["b"] not in graph.nodes or not e["a"] < e["b"]:
                raise ValueError(f"edge {e['a']!r} - {e['b']!r} is malformed or points at a missing node")
            graph.edges[(e["a"], e["b"])] = Edge(e["a"], e["b"], dict(e["mentions"]), e.get("summary"))
        return graph

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "KnowledgeGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
