"""r01: The two baselines, keyword search and vector-only search.

Motto: before building something clever, measure something simple.

BM25 scores documents by shared words (rare words count more, long documents count less).
The "naive" vector search compares fixed-size vectors made by a fake hashing embedder.
Both are scored on the question list: recall@k (how many of the right documents are in
the top k) and MRR (how near the top the first right document is).

Run:  python r01_baselines/code.py --offline
      python r01_baselines/code.py --query "What does a pre-tool hook do?"
      python r01_baselines/code.py --k 5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import raka_common as rc  # noqa: E402
from raka_search import BM25Index, VectorIndex, full_ranking  # noqa: E402,F401  (moved to a shared module in r02)


def run_eval(k: int = 3) -> dict:
    docs = rc.load_corpus()
    questions = rc.load_questions()
    problems = rc.validate_questions(questions, docs)
    if problems:
        raise ValueError("question file problems: " + "; ".join(problems))
    results = {}
    for name, index in (("bm25", BM25Index(docs)), ("vector", VectorIndex(docs))):
        results[name] = rc.evaluate(full_ranking(index, docs), questions, k)
    return results


def print_report(results: dict, k: int) -> None:
    print(f"\nPer question (top {k}; * = a gold document)")
    bm, vec = results["bm25"]["rows"], results["vector"]["rows"]
    for a, b in zip(bm, vec):
        def show(row):
            return ",".join(("*" if t in row["gold"] else "") + t.split("_")[0] for t in row["top"])
        print(f"  {a['id']} {a['type']:<6} gold={','.join(g.split('_')[0] for g in a['gold']):<10} "
              f"bm25=[{show(a)}] vector=[{show(b)}]")
    print(f"\nSummary (recall@{k} and MRR; 1.00 is perfect)")
    print(f"  {'method':<8}{'group':<8}{'n':>3}{'recall':>9}{'mrr':>7}")
    for name in ("bm25", "vector"):
        for group in ("all", "local", "global"):
            s = results[name]["summary"][group]
            print(f"  {name:<8}{group:<8}{s['n']:>3}{s['recall']:>9.2f}{s['mrr']:>7.2f}")
    print("\nNote: 10 short documents is a toy corpus. The point is a working scoring harness; "
          "the real comparison needs a bigger corpus (r07).")


def main(argv: list[str]) -> int:
    k = 3
    query = None
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--k" and args:
            k = int(args.pop(0))
        elif arg == "--query" and args:
            query = args.pop(0)
        elif arg == "--offline":
            continue  # accepted for parity with the Bedrock course; r01 is always offline
        else:
            print(f"unknown argument: {arg}")
            return 2
    if query:
        docs = rc.load_corpus()
        for name, index in (("bm25", BM25Index(docs)), ("vector", VectorIndex(docs))):
            hits = ", ".join(f"{i} ({s:.2f})" for i, s in index.search(query, k))
            print(f"{name:<7}{hits or 'no match'}")
        return 0
    results = run_eval(k)
    print("rakaRAG r01: keyword (BM25) and vector baselines, offline")
    print_report(results, k)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
