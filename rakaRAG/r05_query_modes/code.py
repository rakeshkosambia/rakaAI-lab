"""r05: Query modes: naive, local, global, hybrid, mix.

Motto: a graph only earns its cost if it finds what plain search misses.

Builds the r04 graph, then answers each test question five ways and scores how well the top
chunks cover the gold documents, how many words go into the prompt, and how many extra model
calls the mode costs. No answer is generated here (that needs a real model, r08): this chapter
measures RETRIEVAL.

Run:  python r05_query_modes/code.py --offline
      python r05_query_modes/code.py --rerank
      python r05_query_modes/code.py --ask "Which mechanisms keep the context window small?" --mode hybrid
      python r05_query_modes/code.py --demo-faults
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import raka_chunking as ck  # noqa: E402
import raka_common as rc  # noqa: E402
import raka_extract as ex  # noqa: E402
from raka_fake_model import KEYWORD_FAULTS, FakeExtractor, FakeKeywordModel  # noqa: E402
from raka_graph import KnowledgeGraph  # noqa: E402
from raka_retrieve import MODES, Retriever, graph_context_words, render_context  # noqa: E402


def build(keyword_model=None, strategy: str = "recursive", size: int = 40, overlap: int = 0):
    chunks = ck.chunk_corpus(rc.load_corpus(), strategy, size, overlap)
    graph = KnowledgeGraph()
    graph.add_all(ex.extract_corpus(chunks, FakeExtractor()))
    return chunks, graph, Retriever(chunks, graph, keyword_model or FakeKeywordModel())


def evaluate_mode(retriever: Retriever, mode: str, questions: list[dict], k: int, rerank: bool = False) -> dict:
    cache: dict[str, object] = {}

    def context(question: str):
        if question not in cache:
            cache[question] = retriever.retrieve(question, mode, rerank=rerank)
        return cache[question]

    result = rc.evaluate_chunk_retrieval(
        lambda q: context(q).chunks, questions, k, extra_words=lambda q: graph_context_words(context(q))
    )
    result["calls"] = sum(context(q["question"]).keyword_calls for q in questions) / len(questions)
    result["fallbacks"] = sum(1 for q in questions if context(q["question"]).fallback)
    return result


def run_eval(k: int, rerank: bool = False) -> dict:
    docs = rc.load_corpus()
    questions = rc.load_questions()
    problems = rc.validate_questions(questions, docs)
    if problems:
        raise ValueError("question file problems: " + "; ".join(problems))
    _, _, retriever = build()
    return {mode: evaluate_mode(retriever, mode, questions, k, rerank) for mode in MODES}


def print_table(results: dict, k: int, title: str) -> None:
    print(f"\n{title} (top {k} chunks)")
    print(f"  {'mode':<8}{'cov all':>8}{'local q':>8}{'global q':>9}{'mrr':>6}{'words':>7}{'calls/q':>8}")
    for mode in MODES:
        r = results[mode]
        s = r["summary"]
        print(f"  {mode:<8}{s['all']['coverage']:>8.2f}{s['local']['coverage']:>8.2f}{s['global']['coverage']:>9.2f}{s['all']['mrr']:>6.2f}{s['all']['words']:>7.0f}{r['calls']:>8.0f}")


def print_report(rerank_only: bool = False) -> None:
    ks = (3, 6)
    if not rerank_only:
        for k in ks:
            print_table(run_eval(k), k, "Ordered by graph votes (how many selected entities and relations cite a chunk)")
    for k in ks:
        print_table(run_eval(k, rerank=True), k, "Same candidate pool, re-ordered by keyword score to the question")
    print("\ncov = share of the gold documents reached; local q / global q = by question type; words = prompt size "
          "(chunks + entity and relation text); calls/q = extra model calls per question.")
    print("Test set caution: 12 questions on 10 short documents where plain search already scores about 1.00. "
          "This cannot show a graph winning; r07 adds harder questions.")


def ask(question: str, mode: str, rerank: bool) -> int:
    _, graph, retriever = build()
    ctx = retriever.retrieve(question, mode, rerank=rerank)
    print(f"mode={mode} rerank={rerank} fallback={ctx.fallback} keyword calls={ctx.keyword_calls}")
    print(f"low-level keywords:  {ctx.low}")
    print(f"high-level keywords: {ctx.high}")
    print(f"pool: {len(ctx.chunks)} chunks, {len(ctx.entities)} entities, {len(ctx.relations)} relations\n")
    print(render_context(graph, ctx, k_chunks=4))
    return 0


def demo_faults() -> int:
    question = "Which mechanisms keep the context window small?"
    print(f"Question: {question}\n")
    print(f"{'keyword model fault':<22}{'fallback used':>14}  top 3 chunks (hybrid)")
    for fault in (None,) + tuple(KEYWORD_FAULTS):
        _, _, retriever = build(FakeKeywordModel(fault))
        ctx = retriever.retrieve(question, "hybrid")
        print(f"{fault or 'none':<22}{str(ctx.fallback):>14}  {', '.join(c.id for c in ctx.chunks[:3])}")
    return 0


def main(argv: list[str]) -> int:
    args = list(argv)
    mode, rerank, question, demo = "hybrid", False, None, False
    try:
        while args:
            arg = args.pop(0)
            if arg == "--ask" and args:
                question = args.pop(0)
            elif arg == "--mode" and args:
                mode = args.pop(0)
            elif arg == "--rerank":
                rerank = True
            elif arg == "--demo-faults":
                demo = True
            elif arg == "--offline":
                continue  # accepted for parity with the Bedrock course; r05 is always offline
            else:
                print(f"unknown argument: {arg}")
                return 2
        if demo:
            return demo_faults()
        if question:
            return ask(question, mode, rerank)
    except ValueError as err:
        print(f"error: {err}")
        return 2
    print("rakaRAG r05: query modes with a fake extraction model and a fake keyword model, offline")
    print_report(rerank_only=rerank)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
