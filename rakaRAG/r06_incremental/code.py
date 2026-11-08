"""r06: Incremental add, update and delete.

Motto: a change to one document should cost one document, and leave nothing stale behind.

The index keeps documents, chunks, a cache of the model's extraction answers, and the graph. Adding
a document costs one model call per chunk. Deleting costs none: the graph is derived from per-chunk
mentions, so we remove that document's mentions and drop what is empty. Updating re-uses the cache
for every chunk whose text did not change. `verify` is the doctor check: every cross-reference
holds and the graph equals a fresh rebuild from the cache.

Run:  python r06_incremental/code.py --offline
      python r06_incremental/code.py --delete 04_hooks
      python r06_incremental/code.py --demo-update
      python r06_incremental/code.py --demo-fail
      python r06_incremental/code.py --save
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import raka_common as rc  # noqa: E402
from raka_fake_model import FakeExtractor  # noqa: E402
from raka_store import Index  # noqa: E402

DEFAULT_SAVE = ROOT / "index" / "rakarag"
ADDED_SENTENCE = " A hook can also send a notification to the user."


def build_index(docs=None, model=None) -> Index:
    index = Index(model or FakeExtractor())
    for doc in (docs if docs is not None else rc.load_corpus()):
        index.add_document(doc)
    return index


def graph_json(index: Index) -> str:
    return json.dumps(index.graph.to_dict(), sort_keys=True)


def delete_matches_scratch(doc_id: str) -> tuple[dict, bool]:
    """Delete one document from a full index; compare with an index that never saw it."""
    docs = rc.load_corpus()
    full = build_index(docs)
    report = full.delete_document(doc_id)
    scratch = build_index([d for d in docs if d.id != doc_id])
    return report, graph_json(full) == graph_json(scratch) and not full.verify()


def print_report() -> None:
    docs = rc.load_corpus()
    base = build_index(docs)
    print(f"\nFull build: {len(base.docs)} documents, {len(base.chunks)} chunks, {len(base.graph.nodes)} nodes, "
          f"{len(base.graph.edges)} edges, {base.model_calls} model calls.")
    print("\nDelete each document in turn and compare with an index that never saw it:")
    print(f"  {'deleted':<24}{'chunks':>7}{'nodes gone':>11}{'nodes kept':>11}{'edges gone':>11}{'model calls':>12}{'= scratch':>10}")
    for doc in docs:
        report, same = delete_matches_scratch(doc.id)
        print(f"  {doc.id:<24}{report['chunks_removed']:>7}{report['nodes_deleted']:>11}{report['nodes_rebuilt']:>11}"
              f"{report['edges_deleted']:>11}{report['model_calls']:>12}{'yes' if same else 'NO':>10}")
    print("\n'nodes kept' = entities also mentioned by other documents, so they stay with fewer sources. "
          "A full rebuild would cost "
          f"{base.model_calls} model calls every time; a delete costs 0.")


def demo_update() -> int:
    docs = {d.id: d for d in rc.load_corpus()}
    index = build_index(list(docs.values()))
    old = docs["04_hooks"]
    new = rc.Doc(old.id, old.title, old.text.rstrip() + ADDED_SENTENCE)
    result = index.update_document(new)
    print(f"Updated {old.id} (appended one sentence).")
    print(f"  chunks now: {result['chunks']}; model calls: {result['model_calls']}; chunks served from the cache: {result['cache_hits']}")
    print(f"  a full re-extract of the document would have cost {result['chunks']} calls")
    scratch = build_index([new if d == old.id else docs[d] for d in docs])
    print(f"  graph equals a scratch build of the edited corpus: {graph_json(index) == graph_json(scratch)}")
    print(f"  verify: {index.verify() or 'clean'}")
    again = index.update_document(new)
    print(f"  updating again with the same text: {again['status']}, {again['model_calls']} calls")
    return 0


class FailingModel:
    """Wraps a model and raises on the Nth call, to prove a half-finished add leaves nothing behind."""

    def __init__(self, inner, fail_on: int):
        self.inner, self.fail_on, self.calls = inner, fail_on, 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("model unavailable (simulated)")
        return self.inner(prompt)


def demo_fail() -> int:
    docs = rc.load_corpus()
    index = Index(FailingModel(FakeExtractor(), fail_on=7))
    for doc in docs[:1]:
        index.add_document(doc)
    before = graph_json(index)
    print(f"Indexed {docs[0].id}: {len(index.chunks)} chunks. Now adding {docs[1].id}; the model will fail on call 7.")
    try:
        index.add_document(docs[1])
    except RuntimeError as err:
        print(f"  add failed: {err}")
    print(f"  chunks stored: {len(index.chunks)}; {docs[1].id} registered: {docs[1].id in index.docs}")
    print(f"  graph unchanged by the failed add: {graph_json(index) == before}")
    print(f"  verify: {index.verify() or 'clean'}")
    index.model.fail_on = 0  # the outage is over
    retry = index.add_document(docs[1])
    print(f"  retry after the outage: {retry['status']}, {retry['model_calls']} model calls, "
          f"{retry['cache_hits']} chunk(s) served from the cache (answered before the failure) of {retry['chunks']}")
    print(f"  verify: {index.verify() or 'clean'}")
    return 0


def main(argv: list[str]) -> int:
    args = list(argv)
    try:
        while args:
            arg = args.pop(0)
            if arg == "--delete" and args:
                doc_id = args.pop(0)
                if doc_id not in {d.id for d in rc.load_corpus()}:
                    print(f"unknown document {doc_id!r}")
                    return 2
                report, same = delete_matches_scratch(doc_id)
                for key, value in report.items():
                    print(f"  {key:<20}{value}")
                print(f"  equals an index that never saw it: {same}")
                return 0
            if arg == "--demo-update":
                return demo_update()
            if arg == "--demo-fail":
                return demo_fail()
            if arg == "--save":
                path = Path(args.pop(0)) if args else DEFAULT_SAVE
                build_index().save(path)
                print(f"saved {path}")
                loaded = Index.load(path, FakeExtractor())
                print(f"loaded back: {len(loaded.docs)} documents, verify: {loaded.verify() or 'clean'}")
                return 0
            if arg == "--offline":
                continue  # accepted for parity with the Bedrock course; r06 is always offline
            print(f"unknown or incomplete argument: {arg}")
            return 2
    except ValueError as err:
        print(f"error: {err}")
        return 2
    print("rakaRAG r06: incremental add, update and delete, offline")
    print_report()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
