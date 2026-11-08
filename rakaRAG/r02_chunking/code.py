"""r02: Chunking.

Motto: split the text where the meaning splits, and remember where each piece came from.

Splits each document into chunks three ways (whole document, fixed window of words, recursive
split on natural boundaries), searches the chunks with the r01 indexes, and scores how well the
top k chunks cover the gold documents and how many words they would put in a prompt.

Run:  python r02_chunking/code.py --offline
      python r02_chunking/code.py --k 6
      python r02_chunking/code.py --show 04_hooks --strategy recursive --size 30
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import raka_chunking as ck  # noqa: E402
import raka_common as rc  # noqa: E402
from raka_search import BM25Index, VectorIndex  # noqa: E402

# (label, strategy, size in words, overlap in words)
CONFIGS = [
    ("whole document", "doc", 0, 0),
    ("fixed 40/0", "fixed", 40, 0),
    ("fixed 40/10", "fixed", 40, 10),
    ("fixed 20/5", "fixed", 20, 5),
    ("recursive 40", "recursive", 40, 0),
    ("recursive 20", "recursive", 20, 0),
]


def chunk_search(index, by_id: dict):
    """search(question) -> chunks, best first (only chunks that matched)."""

    def search(question: str):
        return [by_id[chunk_id] for chunk_id, _ in index.search(question, k=len(by_id))]

    return search


def run_eval(k: int = 3) -> list[dict]:
    docs = rc.load_corpus()
    questions = rc.load_questions()
    problems = rc.validate_questions(questions, docs)
    if problems:
        raise ValueError("question file problems: " + "; ".join(problems))
    results = []
    for label, strategy, size, overlap in CONFIGS:
        chunks = ck.chunk_corpus(docs, strategy, size, overlap)
        by_id = {c.id: c for c in chunks}
        chunk_docs = ck.as_docs(chunks)
        entry = {
            "label": label,
            "chunks": len(chunks),
            "avg_words": sum(c.words for c in chunks) / len(chunks),
        }
        for name, index in (("bm25", BM25Index(chunk_docs)), ("vector", VectorIndex(chunk_docs))):
            entry[name] = rc.evaluate_chunk_retrieval(chunk_search(index, by_id), questions, k)
        results.append(entry)
    return results


def print_report(results: list[dict], k: int) -> None:
    print(f"\nTop {k} chunks per question. coverage = gold documents reached; words = prompt size (lower is cheaper)")
    print(f"  {'chunking':<16}{'chunks':>7}{'avg w':>7} | {'method':<7}{'cov all':>8}{'cov glob':>9}{'mrr':>6}{'words':>7}")
    for r in results:
        for j, name in enumerate(("bm25", "vector")):
            s = r[name]["summary"]
            head = f"  {r['label']:<16}{r['chunks']:>7}{r['avg_words']:>7.1f} | " if j == 0 else " " * 33 + "| "
            print(f"{head}{name:<7}{s['all']['coverage']:>8.2f}{s['global']['coverage']:>9.2f}{s['all']['mrr']:>6.2f}{s['all']['words']:>7.0f}")
    print("\nNote: with 10 short documents, chunking mostly trades coverage of multi-document questions "
          "for a smaller prompt. A bigger corpus (r07) is needed to judge it fairly.")


def show(doc_id: str, strategy: str, size: int, overlap: int) -> int:
    docs = {d.id: d for d in rc.load_corpus()}
    if doc_id not in docs:
        print(f"unknown document {doc_id!r}; choose one of: {', '.join(sorted(docs))}")
        return 2
    doc = docs[doc_id]
    chunks = ck.chunk_doc(doc, strategy, size, overlap)
    print(f"{doc_id}: {len(chunks)} chunk(s), strategy={strategy} size={size} overlap={overlap}")
    for c in chunks:
        print(f"  [{c.id}] chars {c.start}-{c.end}, {c.words} words: {c.text}")
    return 0


def main(argv: list[str]) -> int:
    k, strategy, size, overlap, show_doc = 3, "recursive", 30, 0, None
    args = list(argv)
    try:
        while args:
            arg = args.pop(0)
            if arg == "--k" and args:
                k = int(args.pop(0))
            elif arg == "--show" and args:
                show_doc = args.pop(0)
            elif arg == "--strategy" and args:
                strategy = args.pop(0)
            elif arg == "--size" and args:
                size = int(args.pop(0))
            elif arg == "--overlap" and args:
                overlap = int(args.pop(0))
            elif arg == "--offline":
                continue  # accepted for parity with the Bedrock course; r02 is always offline
            else:
                print(f"unknown argument: {arg}")
                return 2
        if show_doc:
            return show(show_doc, strategy, size, overlap)
    except ValueError as err:
        print(f"error: {err}")
        return 2
    print("rakaRAG r02: chunking strategies, offline")
    print_report(run_eval(k), k)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
