"""The index: documents, chunks, an extraction cache and the graph, kept consistent. Original code.

Operations
  add_document      chunk -> extract (model, or cache) -> merge into the graph. Atomic: a failure part-way
                    leaves the index exactly as it was.
  delete_document   remove the document's chunks and everything they contributed. No model calls.
  update_document   delete + add. Chunks whose text did not change come from the cache, so only changed
                    text costs model calls.
  verify            the "doctor" check: every cross-reference holds and the graph equals a fresh rebuild
                    from the cache.
  save / load       one folder of JSON files, each written atomically; load refuses an index that fails verify.

The extraction cache is keyed by a hash of (prompt template, row limit, chunk text), so it is reused only
when the same question would be put to the model again. Pure standard library.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import raka_chunking as ck
import raka_common as rc
import raka_extract as ex
from raka_graph import KnowledgeGraph

STORE_VERSION = 1
PROMPT_VERSION = hashlib.sha256(ex.PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:12]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write(path: Path | str, text: str) -> None:
    """Write to a temporary file, then swap it in, so a crash never leaves a half-written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def _strip_summaries(data: dict) -> dict:
    """Summaries come from a model call, so they are not part of the 'derived from mentions' comparison."""
    clean = json.loads(json.dumps(data))
    for item in clean.get("nodes", []) + clean.get("edges", []):
        item["summary"] = None
    return clean


class Index:
    def __init__(self, model, strategy: str = "recursive", size: int = 40, overlap: int = 0,
                 aliases: dict[str, str] | None = None, max_rows: int = 12):
        self.model = model
        self.params = {"strategy": strategy, "size": size, "overlap": overlap, "max_rows": max_rows}
        self.aliases = dict(aliases or {})
        self.docs: dict[str, dict] = {}        # doc_id -> {"hash", "chunks": [ids]}
        self.chunks: dict[str, ck.Chunk] = {}
        self.cache: dict[str, str] = {}        # extraction key -> the model's raw answer
        self.graph = KnowledgeGraph(self.aliases)
        self.model_calls = 0

    # ---------------------------------------------------------------- extraction with cache
    def _key(self, chunk: ck.Chunk) -> str:
        return text_hash(f"{PROMPT_VERSION}\n{self.params['max_rows']}\n{chunk.text}")

    def _extract(self, chunk: ck.Chunk) -> tuple[ex.Extraction, bool]:
        """Return (extraction, was_cached). Only a successful model answer is cached."""
        key = self._key(chunk)
        cached = key in self.cache
        if not cached:
            raw = self.model(ex.build_prompt(chunk.text, self.params["max_rows"]))
            self.model_calls += 1
            self.cache[key] = raw
        return ex.parse_records(self.cache[key], chunk.id), cached

    # ---------------------------------------------------------------- add / delete / update
    def add_document(self, doc: rc.Doc) -> dict:
        digest = text_hash(doc.text)
        if doc.id in self.docs:
            if self.docs[doc.id]["hash"] == digest:
                return {"doc": doc.id, "status": "unchanged", "chunks": len(self.docs[doc.id]["chunks"]),
                        "model_calls": 0, "cache_hits": 0, "rejected": 0}
            raise ValueError(f"{doc.id!r} is already indexed with different text; use update_document")
        chunks = ck.chunk_doc(doc, self.params["strategy"], self.params["size"], self.params["overlap"])
        calls_before, hits, rejected, added = self.model_calls, 0, 0, []
        try:
            for chunk in chunks:
                extraction, cached = self._extract(chunk)
                hits += cached
                rejected += len(extraction.rejected)
                self.graph.add_extraction(extraction)
                self.chunks[chunk.id] = chunk
                added.append(chunk.id)
        except Exception:
            for chunk_id in added:  # roll back: leave the index exactly as it was
                self.graph.remove_chunk(chunk_id)
                del self.chunks[chunk_id]
            raise
        self.docs[doc.id] = {"hash": digest, "chunks": [c.id for c in chunks]}
        return {"doc": doc.id, "status": "added", "chunks": len(chunks), "model_calls": self.model_calls - calls_before,
                "cache_hits": hits, "rejected": rejected}

    def delete_document(self, doc_id: str, drop_cache: bool = False) -> dict:
        if doc_id not in self.docs:
            return {"doc": doc_id, "status": "not_found"}
        gone = {name: set() for name in ("nodes_deleted", "nodes_rebuilt", "edges_deleted", "edges_rebuilt", "edges_orphaned")}
        summaries, dropped_keys = 0, set()
        chunk_ids = list(self.docs[doc_id]["chunks"])
        for chunk_id in chunk_ids:
            step = self.graph.remove_chunk(chunk_id)
            for name in gone:
                gone[name].update(step[name])
            summaries += step["summaries_cleared"]
            dropped_keys.add(self._key(self.chunks[chunk_id]))
            del self.chunks[chunk_id]
        del self.docs[doc_id]
        # a node touched by several of the document's chunks is counted once, and "rebuilt" means it still exists
        edges_deleted = gone["edges_deleted"] | gone["edges_orphaned"]
        cache_dropped = 0
        if drop_cache:
            still_needed = {self._key(c) for c in self.chunks.values()}  # another document may share identical chunk text
            for key in dropped_keys - still_needed:
                cache_dropped += self.cache.pop(key, None) is not None
        return {"doc": doc_id, "status": "deleted", "chunks_removed": len(chunk_ids),
                "nodes_deleted": len(gone["nodes_deleted"]), "nodes_rebuilt": len(gone["nodes_rebuilt"] - gone["nodes_deleted"]),
                "edges_deleted": len(edges_deleted), "edges_rebuilt": len(gone["edges_rebuilt"] - edges_deleted),
                "summaries_cleared": summaries, "model_calls": 0, "cache_dropped": cache_dropped}

    def update_document(self, doc: rc.Doc) -> dict:
        if doc.id in self.docs and self.docs[doc.id]["hash"] == text_hash(doc.text):
            return {"doc": doc.id, "status": "unchanged", "model_calls": 0}
        removed = self.delete_document(doc.id) if doc.id in self.docs else None
        added = self.add_document(doc)
        added["status"] = "updated" if removed else "added"
        added["deleted"] = removed
        return added

    # ---------------------------------------------------------------- the doctor check
    def rebuild_graph(self) -> KnowledgeGraph:
        """A fresh graph built only from the stored chunks and the cache. Makes no model calls."""
        graph = KnowledgeGraph(self.aliases)
        for doc_id in sorted(self.docs):
            for chunk_id in self.docs[doc_id]["chunks"]:
                chunk = self.chunks[chunk_id]
                graph.add_extraction(ex.parse_records(self.cache[self._key(chunk)], chunk.id))
        return graph

    def verify(self) -> list[str]:
        """Return a list of problems; empty means every invariant holds."""
        problems: list[str] = []
        registered = {cid for d in self.docs.values() for cid in d["chunks"]}
        for cid in sorted(registered - set(self.chunks)):
            problems.append(f"document registry names chunk {cid} that is not stored")
        for cid in sorted(set(self.chunks) - registered):
            problems.append(f"chunk {cid} is stored but belongs to no document")
        for key, node in sorted(self.graph.nodes.items()):
            for cid in node.mentions:
                if cid not in self.chunks:
                    problems.append(f"node {key} cites missing chunk {cid}")
        for (a, b), edge in sorted(self.graph.edges.items()):
            if a not in self.graph.nodes or b not in self.graph.nodes:
                problems.append(f"edge {a}|{b} points at a missing node")
            if not edge.mentions:
                problems.append(f"edge {a}|{b} has no mentions")
            for cid in edge.mentions:
                if cid not in self.chunks:
                    problems.append(f"edge {a}|{b} cites missing chunk {cid}")
        if not problems:
            try:
                fresh = _strip_summaries(self.rebuild_graph().to_dict())
            except KeyError as err:
                problems.append(f"the cache lacks an extraction needed to rebuild ({err})")
            else:
                if fresh != _strip_summaries(self.graph.to_dict()):
                    problems.append("the graph differs from a fresh rebuild from the cache")
        return problems

    # ---------------------------------------------------------------- persistence
    def save(self, folder: Path | str) -> None:
        problems = self.verify()
        if problems:
            raise ValueError("refusing to save an inconsistent index: " + "; ".join(problems[:3]))
        folder = Path(folder)
        dump = lambda data: json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True) + "\n"  # noqa: E731
        atomic_write(folder / "chunks.json", dump({cid: {"doc_id": c.doc_id, "index": c.index, "text": c.text, "start": c.start, "end": c.end}
                                                   for cid, c in sorted(self.chunks.items())}))
        atomic_write(folder / "cache.json", dump(self.cache))
        atomic_write(folder / "docs.json", dump(self.docs))
        atomic_write(folder / "graph.json", dump(self.graph.to_dict()))
        atomic_write(folder / "meta.json", dump({"version": STORE_VERSION, "prompt_version": PROMPT_VERSION,
                                                 "params": self.params, "aliases": self.aliases}))

    @classmethod
    def load(cls, folder: Path | str, model) -> "Index":
        folder = Path(folder)
        read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))  # noqa: E731
        try:
            meta = read("meta.json")
            if meta.get("version") != STORE_VERSION:
                raise ValueError(f"not a version {STORE_VERSION} index")
            if meta.get("prompt_version") != PROMPT_VERSION:
                raise ValueError("this index was built with a different extraction prompt; rebuild it")
            p = meta["params"]
            index = cls(model, p["strategy"], p["size"], p["overlap"], meta.get("aliases"), p["max_rows"])
            index.chunks = {cid: ck.Chunk(c["doc_id"], c["index"], c["text"], c["start"], c["end"]) for cid, c in read("chunks.json").items()}
            index.cache = read("cache.json")
            index.docs = read("docs.json")
            index.graph = KnowledgeGraph.load(folder / "graph.json")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as err:
            raise ValueError(f"could not read the index in {folder}: {err}") from None
        problems = index.verify()
        if problems:
            raise ValueError("the saved index is inconsistent: " + "; ".join(problems[:3]))
        return index

    # ---------------------------------------------------------------- retrieval
    def retriever(self, keyword_model=None, **kwargs):
        from raka_retrieve import Retriever
        return Retriever(sorted(self.chunks.values(), key=lambda c: (c.doc_id, c.index)), self.graph, keyword_model, **kwargs)
