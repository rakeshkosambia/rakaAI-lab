"""r07: Evaluation on a bigger, harder test set, with uncertainty and cost.

Motto: write the questions first, say what you expect, then see whether you were right.

24 documents (the 10 originals plus 14 new), 12 original questions plus 27 new ones, half of them
paraphrased so they share few words with their answers. Every method is scored by coverage of the
gold documents, with a random baseline as the floor, a paired bootstrap to say whether a
difference is real, and the cost of indexing and of each query. The test set is frozen by hash and
the predictions in PREDICTIONS.md were written before the first run; the last section scores them.

All models here are offline fakes: a lexical keyword model, a lexical extraction model and a
hashing "embedder". They can show whether the test set separates methods. They cannot judge how a
real model handles paraphrase. That is r08.

Run:  python r07_evaluation/code.py --offline
      python r07_evaluation/code.py --misses bm25
      python r07_evaluation/code.py --frozen
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import raka_chunking as ck  # noqa: E402
import raka_common as rc  # noqa: E402
import raka_eval as ev  # noqa: E402
from raka_fake_model import LEXICON_V2, FakeExtractor, FakeKeywordModel  # noqa: E402
from raka_retrieve import build_keyword_prompt, graph_context_words  # noqa: E402
from raka_search import VectorIndex  # noqa: E402
from raka_store import Index  # noqa: E402

KS = (3, 6)
GRAPH_MODES = ("local", "global", "hybrid", "mix")
METHODS = ("random", "bm25", "vector", "local", "global", "hybrid", "mix", "local+rerank", "hybrid+rerank", "mix+rerank")
GROUPS = ("v1", "keyword", "paraphrase", "new", "all")


class Prepared:
    """Everything built once: corpus, chunks, graph index, retriever and the tagged question list."""

    def __init__(self):
        self.docs = rc.load_corpus_v2()
        self.by_id = {d.id: d for d in self.docs}
        self.questions = [dict(q, set="v1") for q in rc.load_questions()] + [dict(q, set="v2") for q in rc.load_questions(rc.QUESTIONS_V2_FILE)]
        for label, qs in (("v1", rc.load_questions()), ("v2", rc.load_questions(rc.QUESTIONS_V2_FILE))):
            problems = rc.validate_questions(qs, self.docs)
            if problems:
                raise ValueError(f"questions {label}: " + "; ".join(problems))
        self.index = Index(FakeExtractor(lexicon=LEXICON_V2))
        for doc in self.docs:
            self.index.add_document(doc)
        self.chunks = sorted(self.index.chunks.values(), key=lambda c: (c.doc_id, c.index))
        self.by_chunk = {c.id: c for c in self.chunks}
        self.retriever = self.index.retriever(FakeKeywordModel(lexicon=LEXICON_V2))
        self.vector = VectorIndex(ck.as_docs(self.chunks))
        self.random = ev.RandomRetriever(self.chunks)
        self._cache: dict = {}

    def context(self, mode: str, rerank: bool, question: str):
        key = (mode, rerank, question)
        if key not in self._cache:
            self._cache[key] = self.retriever.retrieve(question, mode, rerank=rerank)
        return self._cache[key]

    def method(self, name: str):
        """Return (ranker(question) -> chunks, extra_words(question) -> int, calls(question) -> int)."""
        zero = lambda q: 0  # noqa: E731
        if name == "random":
            return self.random.ranking, zero, zero
        if name == "vector":
            return (lambda q: [self.by_chunk[i] for i, _ in self.vector.search(q, k=len(self.chunks))]), zero, zero
        mode, _, rerank = name.partition("+")
        mode = "naive" if name == "bm25" else mode
        return (lambda q: self.context(mode, bool(rerank), q).chunks,
                lambda q: graph_context_words(self.context(mode, bool(rerank), q)),
                lambda q: self.context(mode, bool(rerank), q).keyword_calls)

    def group_indexes(self) -> dict[str, list[int]]:
        qs = self.questions
        return {
            "v1": [i for i, q in enumerate(qs) if q["set"] == "v1"],
            "keyword": [i for i, q in enumerate(qs) if q["set"] == "v2" and q["style"] == "keyword"],
            "paraphrase": [i for i, q in enumerate(qs) if q["set"] == "v2" and q["style"] == "paraphrase"],
            "new": [i for i, q in enumerate(qs) if q["set"] == "v2"],
            "all": list(range(len(qs))),
        }


def evaluate_all(prep: Prepared) -> dict:
    """results[method][k] = {rows, coverage per group, words, calls}."""
    groups = prep.group_indexes()
    results: dict = {}
    for name in METHODS:
        ranker, extra, calls = prep.method(name)
        results[name] = {}
        for k in KS:
            res = rc.evaluate_chunk_retrieval(ranker, prep.questions, k, extra_words=extra)
            cov = [r["coverage"] for r in res["rows"]]
            results[name][k] = {
                "rows": res["rows"],
                "coverage": {g: sum(cov[i] for i in idx) / len(idx) for g, idx in groups.items()},
                "words": sum(r["words"] for r in res["rows"]) / len(res["rows"]),
                "calls": sum(calls(q["question"]) for q in prep.questions) / len(prep.questions),
            }
    return results


def differences(prep: Prepared, results: dict, k: int, group: str, versus: str = "bm25") -> dict:
    idx = prep.group_indexes()[group]
    base = [results[versus][k]["rows"][i]["coverage"] for i in idx]
    out = {}
    for name in METHODS:
        if name in (versus, "random"):
            continue
        mine = [results[name][k]["rows"][i]["coverage"] for i in idx]
        mean, low, high = ev.paired_bootstrap(mine, base)
        out[name] = {"mean": mean, "low": low, "high": high, "verdict": ev.clear_difference(low, high)}
    return out


def test_set_summary(prep: Prepared) -> dict:
    new = [q for q in prep.questions if q["set"] == "v2"]
    overlap = {}
    for style in rc.QUESTION_STYLES:
        values = [ev.lexical_overlap(q["question"], [prep.by_id[g].text for g in q["gold_docs"]]) for q in new if q["style"] == style]
        overlap[style] = sum(values) / len(values)
    return {"documents": len(prep.docs), "chunks": len(prep.chunks), "v1": sum(1 for q in prep.questions if q["set"] == "v1"),
            "new": len(new), "keyword": sum(1 for q in new if q["style"] == "keyword"),
            "paraphrase": sum(1 for q in new if q["style"] == "paraphrase"),
            "local": sum(1 for q in new if q["type"] == "local"), "global": sum(1 for q in new if q["type"] == "global"),
            "overlap": overlap}


def check_predictions(prep: Prepared, results: dict) -> list[dict]:
    """Score the eight predictions in PREDICTIONS.md against the numbers."""
    cov = lambda m, k, g: results[m][k]["coverage"][g]  # noqa: E731
    out = []

    def add(pid, claim, holds, evidence):
        out.append({"id": pid, "claim": claim, "holds": bool(holds), "evidence": evidence})

    add("P1", "random coverage at k=3 is below 0.15", cov("random", 3, "all") < 0.15, f"{cov('random', 3, 'all'):.2f}")
    kw6, pa6 = cov("bm25", 6, "keyword"), cov("bm25", 6, "paraphrase")
    add("P2", "BM25 at k=6: keyword questions >= 0.80 and paraphrase questions <= 0.60", kw6 >= 0.80 and pa6 <= 0.60, f"keyword {kw6:.2f}, paraphrase {pa6:.2f}")
    v6 = cov("vector", 6, "paraphrase")
    add("P3", "fake vector search is no better than BM25 on paraphrase questions (k=6)", v6 <= pa6, f"vector {v6:.2f} vs bm25 {pa6:.2f}")
    diffs = differences(prep, results, 6, "paraphrase")
    higher = [m for m in GRAPH_MODES if diffs[m]["verdict"] == "higher"]
    add("P4", "no graph mode is clearly above BM25 on paraphrase questions (k=6, 95% interval)", not higher,
        ", ".join(f"{m} {diffs[m]['mean']:+.2f} [{diffs[m]['low']:+.2f},{diffs[m]['high']:+.2f}]" for m in GRAPH_MODES))
    base_words = results["bm25"][3]["words"]
    ratios = {m: results[m][3]["words"] / base_words for m in ("local", "hybrid", "mix")}
    add("P5", "local, hybrid and mix cost 5 to 9 times BM25's prompt words (k=3) and 1 model call per question",
        all(5 <= r <= 9 for r in ratios.values()) and all(results[m][3]["calls"] == 1 for m in GRAPH_MODES),
        ", ".join(f"{m} {r:.1f}x" for m, r in ratios.items()))
    v1 = cov("bm25", 3, "v1")
    add("P6", "BM25 on the original 12 questions at k=3 stays at 0.85 or more", v1 >= 0.85, f"{v1:.2f} (was 0.97 with 10 documents)")
    rr = cov("hybrid+rerank", 6, "all")
    base_all = cov("bm25", 6, "all")
    add("P7", "re-ordered hybrid is within 0.10 of BM25 and not above it by more than 0.05 (k=6, all)",
        abs(rr - base_all) <= 0.10 and rr <= base_all + 0.05, f"hybrid+rerank {rr:.2f} vs bm25 {base_all:.2f}")
    widths = [diffs[m]["high"] - diffs[m]["low"] for m in GRAPH_MODES]
    mean_width = sum(widths) / len(widths)
    add("P8", "the 95% intervals on paraphrase questions are about 0.08 to 0.30 wide", 0.08 <= mean_width <= 0.30, f"mean width {mean_width:.2f}")
    return out


def print_report(prep: Prepared, results: dict) -> None:
    s = test_set_summary(prep)
    frozen = ev.check_frozen()
    print(f"\nTest set: {s['documents']} documents, {s['chunks']} chunks (recursive 40); {s['v1']} original questions + {s['new']} new "
          f"({s['keyword']} keyword-style, {s['paraphrase']} paraphrase-style; {s['local']} local, {s['global']} global).")
    print(f"Frozen test set: {'untouched since it was frozen' if not frozen else 'CHANGED: ' + '; '.join(frozen)}")
    print(f"Mean word overlap between question and answer documents: keyword {s['overlap']['keyword']:.2f}, paraphrase {s['overlap']['paraphrase']:.2f} "
          "(1.00 = every question word appears in the answer)")

    for k in KS:
        print(f"\nCoverage of gold documents, top {k} chunks (group sizes: " + ", ".join(f"{g} {len(prep.group_indexes()[g])}" for g in GROUPS) + ")")
        print(f"  {'method':<15}" + "".join(f"{g:>11}" for g in GROUPS) + f"{'words':>8}{'calls/q':>8}")
        for name in METHODS:
            r = results[name][k]
            print(f"  {name:<15}" + "".join(f"{r['coverage'][g]:>11.2f}" for g in GROUPS) + f"{r['words']:>8.0f}{r['calls']:>8.0f}")

    for group, k in (("paraphrase", 6), ("new", 6), ("paraphrase", 3)):
        print(f"\nDifference from BM25 on '{group}' questions, top {k} (paired bootstrap, 95% interval; 'unclear' = cannot tell apart)")
        for name, d in differences(prep, results, k, group).items():
            print(f"  {name:<15}{d['mean']:>+7.2f}  [{d['low']:>+5.2f}, {d['high']:>+5.2f}]  {d['verdict']}")

    cost = ev.indexing_cost(prep.index)
    kw_words = sum(len(build_keyword_prompt(q["question"]).split()) for q in prep.questions) / len(prep.questions)
    print("\nCost")
    print(f"  Building the graph index: {cost['calls']} model calls for {cost['documents']} documents ({cost['chunks']} chunks); "
          f"about {cost['tokens_in']:,} tokens in and {cost['tokens_out']:,} out (estimate: {ev.TOKENS_PER_WORD} tokens per word, no prices assumed).")
    print(f"  Plain search needs no model calls to index. Each graph-mode question adds 1 model call (~{ev.estimate_tokens(kw_words)} tokens in) "
          f"and about {results['hybrid'][3]['words'] - results['bm25'][3]['words']:.0f} extra prompt words at k=3.")
    print("  Editing one document re-extracts only its changed chunks (r06), so index cost scales with change, not corpus size.")

    print("\nPredictions from PREDICTIONS.md (written before the first run)")
    for p in check_predictions(prep, results):
        print(f"  {p['id']} {'HOLDS' if p['holds'] else 'WRONG':<6} {p['claim']}\n         {p['evidence']}")
    print("\nThe offline fakes are lexical, so this run shows that the test set separates methods, not how a real model will behave (r08).")


def show_misses(prep: Prepared, results: dict, method: str, k: int = 6) -> int:
    if method not in METHODS:
        print(f"unknown method {method!r}; choose one of: {', '.join(METHODS)}")
        return 2
    rows = results[method][k]["rows"]
    print(f"{method}: questions that do not reach every gold document in the top {k} chunks")
    for i, r in enumerate(rows):
        if r["coverage"] < 1.0:
            q = prep.questions[i]
            print(f"  {q['id']} [{q['set']}/{q.get('style', '-')}/{q['type']}] coverage {r['coverage']:.2f}: {q['question']}")
    return 0


def main(argv: list[str]) -> int:
    args = list(argv)
    miss = None
    try:
        while args:
            arg = args.pop(0)
            if arg == "--misses" and args:
                miss = args.pop(0)
            elif arg == "--frozen":
                problems = ev.check_frozen()
                print("test set untouched since it was frozen" if not problems else "CHANGED: " + "; ".join(problems))
                return 1 if problems else 0
            elif arg == "--offline":
                continue  # accepted for parity with the Bedrock course; r07 is always offline
            else:
                print(f"unknown or incomplete argument: {arg}")
                return 2
    except ValueError as err:
        print(f"error: {err}")
        return 2
    prep = Prepared()
    results = evaluate_all(prep)
    if miss:
        return show_misses(prep, results, miss)
    print("rakaRAG r07: evaluation on a bigger, harder test set (offline fakes)")
    print_report(prep, results)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
