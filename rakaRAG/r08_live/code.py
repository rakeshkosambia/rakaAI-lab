"""r08: A live proof of concept on Amazon Bedrock under a hard token cap.

Motto: spend a little, measure honestly, stop when the money says stop.

Question: do real embeddings and a real extraction model change the picture r07 painted with fakes?
Pipeline, in priority order (cheapest and most informative first), every call through one budget guard:

  smoke      one tiny embedding and one tiny text call, to catch request-shape mistakes (~60 tokens)
  embed      embed the 24 documents and the 33 retrieval questions with the embedding model
  extract    one cheap-model call per document builds the knowledge graph (whole-document chunks)
  keywords   one cheap-model call per question for the 15 new questions (graph modes)
  answers    a stronger model answers one question twice (vector search vs graph + vector), with citations
             that WE check, not the model

If the cap would be crossed, the stage in progress stops and the report says what is missing; nothing is
invented. The offline mode runs the identical code against a fake, costs nothing, and proves the wiring.

Run:  python r08_live/code.py --plan                 # token plan, no AWS needed
      python r08_live/code.py --offline              # full pipeline against the offline fake
      python r08_live/code.py --live --smoke         # ~60 tokens: do the request shapes work for YOUR models?
      python r08_live/code.py --live --yes           # the real run, capped (default 20,000 tokens)
      python r08_live/code.py --report               # print the saved results
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import raka_answer as an  # noqa: E402
import raka_bedrock as bd  # noqa: E402
import raka_chunking as ck  # noqa: E402
import raka_common as rc  # noqa: E402
import raka_eval as ev  # noqa: E402
import raka_extract as ex  # noqa: E402
from raka_graph import KnowledgeGraph  # noqa: E402
from raka_retrieve import Retriever, build_keyword_prompt, graph_context_words  # noqa: E402
from raka_store import Index  # noqa: E402

RESULTS_FILE = ROOT / "index" / "r08_results.json"
MAX_OUT = {"extract": 220, "keywords": 80, "answers": 220, "smoke": 10}
EXTRACT_ROWS = 8                       # fewer rows per answer keeps the output (and the bill) small
KS = (3, 5)                            # top documents; chunks are whole documents here
ANSWER_SAMPLE = ("b01", "b10", "b15")  # which questions get a written answer, in order
ANSWER_METHODS = ("vector", "mix_vec")
PLAIN_METHODS = ("random", "bm25", "vector")
GRAPH_METHODS = ("local", "global", "hybrid", "mix_vec", "hybrid>vec")
METHODS = PLAIN_METHODS + GRAPH_METHODS
GROUPS = ("v3", "v2p", "all")


class Data:
    """The documents (as whole-document chunks) and the 33 retrieval questions, tagged by set."""

    def __init__(self):
        self.docs = rc.load_corpus_v2()
        self.by_id = {d.id: d for d in self.docs}
        self.chunks = ck.chunk_corpus(self.docs, "doc")
        self.by_chunk = {c.id: c for c in self.chunks}
        v3 = rc.load_questions(ev.QUESTIONS_V3_FILE)
        v2p = [q for q in rc.load_questions(rc.QUESTIONS_V2_FILE) if q["style"] == "paraphrase"]
        for label, qs in (("v3", v3), ("v2p", v2p)):
            problems = rc.validate_questions(qs, self.docs)
            if problems:
                raise ValueError(f"questions {label}: " + "; ".join(problems))
        self.questions = [dict(q, set="v3") for q in v3] + [dict(q, set="v2p") for q in v2p]
        self.v3 = [q for q in self.questions if q["set"] == "v3"]


# ---------------------------------------------------------------- the plan (no AWS, no tokens)
def make_plan(data: Data, answers: int = 1) -> dict:
    """Likely and worst-case tokens per stage, computed from the real prompt sizes."""
    likely, worst = bd.ACTUAL_TOKENS_PER_WORD, bd.reserve_tokens
    stages = []

    def stage(name, calls, likely_tokens, worst_tokens, note):
        stages.append({"stage": name, "calls": calls, "likely": math.ceil(likely_tokens), "worst": math.ceil(worst_tokens), "note": note})

    doc_words = [bd.words(c.text) for c in data.chunks]
    q_words = [bd.words(q["question"]) for q in data.questions]
    stage("smoke", 2, 60, 60, "one tiny embedding, one tiny text call")
    stage("embed", len(doc_words) + len(q_words), sum(w * likely for w in doc_words + q_words),
          sum(worst(w) for w in doc_words + q_words), "24 documents + 33 questions")
    prompts = [bd.words(ex.build_prompt(c.text, EXTRACT_ROWS)) for c in data.chunks]
    stage("extract", len(prompts), sum(p * likely + 0.6 * MAX_OUT["extract"] for p in prompts),
          sum(worst(p, MAX_OUT["extract"]) for p in prompts), f"one call per document, at most {MAX_OUT['extract']} output tokens each")
    kw = [bd.words(build_keyword_prompt(q["question"])) for q in data.v3]
    stage("keywords", len(kw), sum(p * likely + 0.5 * MAX_OUT["keywords"] for p in kw),
          sum(worst(p, MAX_OUT["keywords"]) for p in kw), "15 new questions, cheap model")
    n_calls = answers * len(ANSWER_METHODS)
    stage("answers", n_calls, n_calls * (430 * likely + 100 * 1.3 + 0.6 * MAX_OUT["answers"] + 100),
          n_calls * worst(520, MAX_OUT["answers"]), f"{answers} question(s) x {len(ANSWER_METHODS)} methods, stronger model, about 3 documents of context")
    return {"stages": stages, "likely_total": sum(s["likely"] for s in stages), "worst_total": sum(s["worst"] for s in stages)}


def print_plan(plan: dict, cap: int) -> None:
    print(f"\nToken plan (input + output, every call counted). Cap: {cap:,}.")
    print(f"  {'stage':<10}{'calls':>6}{'likely':>9}{'worst':>9}  note")
    for s in plan["stages"]:
        print(f"  {s['stage']:<10}{s['calls']:>6}{s['likely']:>9,}{s['worst']:>9,}  {s['note']}")
    print(f"  {'total':<10}{sum(s['calls'] for s in plan['stages']):>6}{plan['likely_total']:>9,}{plan['worst_total']:>9,}")
    verdict = "fits" if plan["likely_total"] <= cap else "does NOT fit"
    print(f"\n  Likely total {plan['likely_total']:,} {verdict} under the cap ({plan['likely_total'] / cap:.0%} of it).")
    if plan["worst_total"] > cap:
        print(f"  Worst case {plan['worst_total']:,} is above the cap: the guard refuses any single call whose own worst case would cross it, "
              "so the last stage (answers) is the one that gets cut first.")
    print("  Tokens are estimates here (1.3 per word likely, 1.6 per word worst case); the live run counts the real figures the service reports.")


# ---------------------------------------------------------------- the pipeline
def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def run_pipeline(bedrock: bd.Bedrock, data: Data, answers: int = 1) -> dict:
    cfg, budget = bedrock.config, bedrock.budget
    res: dict = {"version": 1, "mode": "offline" if cfg.offline else "live",
                 "models": {"extract": cfg.extract_model, "query": cfg.query_model, "embed": cfg.embed_model, "region": cfg.region},
                 "complete": {}, "stopped": None, "notes": []}

    def stopped(stage: str, exc: Exception):
        res["stopped"] = {"stage": stage, "reason": str(exc)}

    worst = {s["stage"]: s["worst"] for s in make_plan(data, answers)["stages"]}

    def fits(stage: str) -> bool:
        """A stage that builds one thing from many calls (the graph, the keywords) starts only if ALL of it fits in the
        worst case, so the cap can never leave a half-built, unusable result behind."""
        if budget.remaining >= worst[stage]:
            return True
        stopped(stage, bd.BudgetExceeded(stage, "pre-check", worst[stage], budget.remaining))
        return False

    # ---- stage: embed
    doc_vecs: dict[str, list[float]] = {}
    q_vecs: dict[str, list[float]] = {}
    try:
        for c in data.chunks:
            doc_vecs[c.id] = bedrock.embed(c.text, "document", "embed", c.doc_id)
        for q in data.questions:
            q_vecs[q["id"]] = bedrock.embed(q["question"], "query", "embed", q["id"])
        res["complete"]["embed"] = True
    except (bd.BudgetExceeded, bd.LiveCallError) as exc:
        res["complete"]["embed"] = False
        stopped("embed", exc)
    vectors_ok = res["complete"]["embed"]

    # ---- stage: extract (the graph)
    index = Index(lambda prompt: bedrock.converse(cfg.extract_model, prompt, MAX_OUT["extract"], "extract", f"doc{len(index.docs) + 1}"),
                  strategy="doc", max_rows=EXTRACT_ROWS)
    if res["stopped"] is None and fits("extract"):
        try:
            for doc in data.docs:
                index.add_document(doc)
            res["complete"]["extract"] = True
        except (bd.BudgetExceeded, bd.LiveCallError) as exc:
            res["complete"]["extract"] = False
            stopped("extract", exc)
    else:
        res["complete"]["extract"] = False
    graph_ok = res["complete"]["extract"]
    rows = ents = rels = rejected = complete_marker = 0
    for chunk in index.chunks.values():
        parsed = ex.parse_records(index.cache[index._key(chunk)], chunk.id)
        ents += len(parsed.entities)
        rels += len(parsed.relations)
        rejected += len(parsed.rejected)
        complete_marker += parsed.complete
    res["extraction"] = {"documents_done": len(index.docs), "documents": len(data.docs), "entities": ents, "relations": rels,
                         "rejected_rows": rejected, "with_completion_marker": complete_marker,
                         "nodes": len(index.graph.nodes), "edges": len(index.graph.edges)}

    # ---- retrievers
    def vec_rank(question_id: str) -> list[str]:
        order = sorted(doc_vecs, key=lambda cid: (-cosine(q_vecs[question_id], doc_vecs[cid]), cid))
        return order

    qid_of = {q["question"]: q["id"] for q in data.questions}
    plain = Retriever(data.chunks, KnowledgeGraph(), None)
    kw_cache: dict[str, str] = {}
    fallbacks = {"valid": 0, "fallback": 0}

    def kw_model(prompt: str) -> str:
        if prompt not in kw_cache:
            kw_cache[prompt] = bedrock.converse(cfg.extract_model, prompt, MAX_OUT["keywords"], "keywords", f"kw{len(kw_cache) + 1}")
        return kw_cache[prompt]

    graph_r = vec_r = None
    if graph_ok and vectors_ok and fits("keywords"):
        graph_r = Retriever(data.chunks, index.graph, kw_model)
        vec_r = Retriever(data.chunks, index.graph, kw_model, chunk_search=lambda q, k: vec_rank(qid_of[q])[:k])
        try:
            for q in data.v3:  # one keyword call per question; memoised for every graph mode afterwards
                ctx = graph_r.retrieve(q["question"], "hybrid")
                fallbacks["fallback" if ctx.fallback else "valid"] += 1
            res["complete"]["keywords"] = True
        except (bd.BudgetExceeded, bd.LiveCallError) as exc:
            res["complete"]["keywords"] = False
            stopped("keywords", exc)
    else:
        res["complete"]["keywords"] = False
    res["keywords"] = dict(fallbacks, total=len(data.v3))
    graph_ready = bool(res["complete"]["keywords"])

    # ---- retrieval scores (no tokens spent here)
    rankers = {"random": lambda q: ev.RandomRetriever(data.chunks).ranking(q["question"]),
               "bm25": lambda q: plain.retrieve(q["question"], "naive").chunks}
    if vectors_ok:
        rankers["vector"] = lambda q: [data.by_chunk[cid] for cid in vec_rank(q["id"])]
    if graph_ready:
        rankers.update({
            "local": lambda q: graph_r.retrieve(q["question"], "local").chunks,
            "global": lambda q: graph_r.retrieve(q["question"], "global").chunks,
            "hybrid": lambda q: graph_r.retrieve(q["question"], "hybrid").chunks,
            "mix_vec": lambda q: vec_r.retrieve(q["question"], "mix").chunks,
            "hybrid>vec": lambda q: sorted(graph_r.retrieve(q["question"], "hybrid").chunks, key=lambda c: (-cosine(q_vecs[q["id"]], doc_vecs[c.id]), c.id)),
        })
    retrieval: dict = {}
    per_question: dict = {}
    for name, ranker in rankers.items():
        qs = data.questions if name in PLAIN_METHODS else data.v3
        retrieval[name] = {}
        for k in KS:
            by_q = {}
            for q in qs:
                ranked = ranker(q)[:k]
                by_q[q["id"]] = len({c.doc_id for c in ranked} & set(q["gold_docs"])) / len(q["gold_docs"])
            per_question[(name, k)] = by_q
            groups = {"v3": [q["id"] for q in data.v3]}
            if name in PLAIN_METHODS:
                groups["v2p"] = [q["id"] for q in data.questions if q["set"] == "v2p"]
                groups["all"] = [q["id"] for q in data.questions]
            retrieval[name][str(k)] = {g: sum(by_q[i] for i in ids) / len(ids) for g, ids in groups.items()}
    res["retrieval"] = retrieval

    def diff(name: str, versus: str, k: int, ids: list[str]) -> dict | None:
        if (name, k) not in per_question or (versus, k) not in per_question:
            return None
        a, b = per_question[(name, k)], per_question[(versus, k)]
        mean, low, high = ev.paired_bootstrap([a[i] for i in ids], [b[i] for i in ids])
        return {"mean": mean, "low": low, "high": high, "verdict": ev.clear_difference(low, high)}

    v3_ids = [q["id"] for q in data.v3]
    all_ids = [q["id"] for q in data.questions]
    res["differences"] = {
        "vs_bm25": {m: {str(k): diff(m, "bm25", k, v3_ids) for k in KS} for m in ("vector",) + GRAPH_METHODS if m in rankers},
        "vs_vector": {m: {str(k): diff(m, "vector", k, v3_ids) for k in KS} for m in ("bm25",) + GRAPH_METHODS if m in rankers},
        "vector_vs_bm25_all": {str(k): diff("vector", "bm25", k, all_ids) for k in KS} if vectors_ok else None,
    }

    # ---- stage: answers (the stronger model; we check the citations ourselves)
    res["answers"] = []
    if res["stopped"] is None and graph_ready:
        by_qid = {q["id"]: q for q in data.questions}
        try:
            for qid in ANSWER_SAMPLE[:answers]:
                q = by_qid[qid]
                for method in ANSWER_METHODS:
                    if method == "vector":
                        chunks, lines = [data.by_chunk[cid] for cid in vec_rank(qid)[:3]], []
                    else:
                        ctx = vec_r.retrieve(q["question"], "mix")
                        chunks = ctx.chunks[:3]
                        lines = [f"{n.name} ({n.type}): {n.description[:160]}" for n in ctx.entities[:3]]
                        lines += [f"{index.graph.nodes[e.a].name} -- {index.graph.nodes[e.b].name} [{', '.join(e.keywords)}]: {e.description[:160]}" for e in ctx.relations[:3]]
                    prompt = an.build_answer_prompt(q["question"], [(c.doc_id, c.text) for c in chunks], lines)
                    text = bedrock.converse(cfg.query_model, prompt, MAX_OUT["answers"], "answers", f"{qid}/{method}")
                    check = an.validate_answer(text, [c.doc_id for c in chunks])
                    res["answers"].append({"question": qid, "method": method, "text": text, "provided": [c.doc_id for c in chunks],
                                           "valid": check["ok"], "cited": check["cited"], "problems": check["problems"],
                                           "gold_cited": an.gold_cited(check["cited"], q["gold_docs"])})
            res["complete"]["answers"] = True
        except (bd.BudgetExceeded, bd.LiveCallError) as exc:
            res["complete"]["answers"] = False
            stopped("answers", exc)
    else:
        res["complete"]["answers"] = False

    res["budget"] = budget.to_dict()
    res["predictions"] = check_predictions(res) if res["mode"] == "live" else []
    return res


# ---------------------------------------------------------------- predictions
def check_predictions(res: dict) -> list[dict]:
    """Score L1 to L7 from PREDICTIONS.md. Only meaningful for a live run."""
    out = []

    def add(pid, claim, holds, evidence):
        out.append({"id": pid, "claim": claim, "holds": holds, "evidence": evidence})

    r, d = res["retrieval"], res["differences"]
    have = lambda m: m in r  # noqa: E731
    if have("vector") and have("bm25"):
        v, b = r["vector"]["3"]["v3"], r["bm25"]["3"]["v3"]
        add("L1", "real vector search reaches at least as many gold documents as BM25 on the 15 new questions (k=3)", v >= b, f"vector {v:.2f}, bm25 {b:.2f}")
        add("L2", "BM25 does worse on the new 15 than on the 18 v2 paraphrase questions (k=3)", b < r["bm25"]["3"]["v2p"], f"new {b:.2f} vs v2 {r['bm25']['3']['v2p']:.2f}")
    else:
        add("L1", "real vector search at least as good as BM25 (k=3)", None, "not evaluated: embeddings incomplete")
        add("L2", "BM25 does worse on the new 15 than on the v2 paraphrase questions", None, "not evaluated: embeddings incomplete")
    graph = [m for m in ("local", "global", "hybrid") if m in d["vs_vector"]]
    if graph:
        higher = [m for m in graph if d["vs_vector"][m]["3"]["verdict"] == "higher"]
        add("L3", "no graph mode clearly above real vector search on the new 15 (k=3)", not higher,
            ", ".join(f"{m} {d['vs_vector'][m]['3']['mean']:+.2f} [{d['vs_vector'][m]['3']['low']:+.2f},{d['vs_vector'][m]['3']['high']:+.2f}]" for m in graph))
    else:
        add("L3", "no graph mode clearly above real vector search", None, "not evaluated: graph or keywords incomplete")
    k = res["keywords"]
    add("L4", "at least 13 of 15 keyword answers are usable JSON", (k["valid"] >= 13) if res["complete"].get("keywords") else None, f"{k['valid']} valid, {k['fallback']} fell back of {k['total']}")
    e = res["extraction"]
    total_rows = e["entities"] + e["relations"] + e["rejected_rows"]
    share = e["rejected_rows"] / total_rows if total_rows else 1.0
    add("L5", "at most 20% of extraction rows rejected and at least 20 of 24 documents with the completion marker",
        (share <= 0.20 and e["with_completion_marker"] >= 20) if res["complete"].get("extract") else None,
        f"{share:.0%} rejected, {e['with_completion_marker']} of {e['documents']} complete")
    used = res["budget"]["used"]
    add("L6", "the run uses between 12,000 and 20,000 tokens and never exceeds the cap", 12_000 <= used <= res["budget"]["cap"], f"{used:,} of {res['budget']['cap']:,}")
    valid = [a for a in res["answers"] if a["valid"]]
    add("L7", "at least 2 sampled answers pass the citation check", (len(valid) >= 2) if res["answers"] else None, f"{len(valid)} of {len(res['answers'])} passed")
    return out


# ---------------------------------------------------------------- the report
def print_report(res: dict, plan: dict | None = None) -> None:
    b = res["budget"]
    label = "LIVE (real Bedrock calls)" if res["mode"] == "live" else "OFFLINE (a fake: the numbers below are NOT real results)"
    print(f"\nr08 results: {label}")
    print(f"  models: extraction {res['models']['extract']}, answers {res['models']['query']}, embeddings {res['models']['embed']} ({res['models']['region']})")
    print(f"  tokens used {b['used']:,} of cap {b['cap']:,} ({b['used'] / b['cap']:.0%}); {b['calls']} calls")
    for stage, s in b["by_stage"].items():
        print(f"    {stage:<10}{s['calls']:>4} calls{s['total']:>9,} tokens ({s['in']:,} in, {s['out']:,} out)")
    done = ", ".join(f"{k} {'done' if v else 'NOT done'}" for k, v in res["complete"].items())
    print(f"  stages: {done}")
    if res["stopped"]:
        print(f"  STOPPED in '{res['stopped']['stage']}': {res['stopped']['reason']}")
    e = res["extraction"]
    print(f"  graph: {e['documents_done']} of {e['documents']} documents extracted; {e['nodes']} nodes, {e['edges']} edges; "
          f"{e['rejected_rows']} rows rejected; {e['with_completion_marker']} answers had the completion marker")
    k = res["keywords"]
    print(f"  keyword answers: {k['valid']} usable, {k['fallback']} fell back (of {k['total']})")

    print("\nCoverage of gold documents (top k whole documents). v3 = the 15 OODA-concept questions; v2p = 18 v2 paraphrase questions")
    print(f"  {'method':<12}" + "".join(f"{'k=' + str(k) + ' ' + g:>11}" for k in KS for g in ("v3", "v2p", "all")))
    for m in METHODS:
        if m not in res["retrieval"]:
            continue
        cells = []
        for kk in KS:
            for g in ("v3", "v2p", "all"):
                v = res["retrieval"][m][str(kk)].get(g)
                cells.append(f"{v:>11.2f}" if v is not None else f"{'-':>11}")
        print(f"  {m:<12}" + "".join(cells))
    print("  ('-' = not evaluated: graph methods run on the 15 new questions only, to save tokens)")

    print("\nDifference on the 15 new questions, paired bootstrap 95% interval ('unclear' = cannot tell apart)")
    for versus, key in (("vector", "vs_vector"), ("BM25", "vs_bm25")):
        for m, by_k in res["differences"][key].items():
            for kk in KS:
                x = by_k[str(kk)]
                if x:
                    print(f"  {m:<12} minus {versus:<7} k={kk}: {x['mean']:>+6.2f}  [{x['low']:>+5.2f}, {x['high']:>+5.2f}]  {x['verdict']}")

    if res["answers"]:
        print("\nAnswers (the citation check is ours, not the model's)")
        for a in res["answers"]:
            print(f"  [{a['question']} / {a['method']}] citations {'OK' if a['valid'] else 'FAILED: ' + '; '.join(a['problems'])}; cited {a['cited']}; share of gold cited {a['gold_cited']:.2f}")
            print("    " + a["text"].strip().replace("\n", "\n    "))
    if res["predictions"]:
        print("\nPredictions from PREDICTIONS.md (written before the live run)")
        for p in res["predictions"]:
            status = "NOT CHECKED" if p["holds"] is None else ("HOLDS" if p["holds"] else "WRONG")
            print(f"  {p['id']} {status:<12} {p['claim']}\n               {p['evidence']}")
    elif res["mode"] == "offline":
        print("\nPredictions are only scored on a live run: an offline fake would make them meaningless.")


# ---------------------------------------------------------------- commands
def save_results(res: dict, path: Path | None = None) -> None:
    path = path or RESULTS_FILE  # looked up at call time, so it can be redirected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def smoke(bedrock: bd.Bedrock) -> int:
    vec = bedrock.embed("hello", "query", "smoke", "embed")
    text = bedrock.converse(bedrock.config.extract_model, "Reply with the single word: ok", MAX_OUT["smoke"], "smoke", "converse")
    print(f"embedding: {len(vec)} numbers (unit length: {abs(math.sqrt(sum(x * x for x in vec)) - 1) < 1e-3})")
    print(f"text call returned: {text.strip()[:60]!r}")
    print(f"tokens used: {bedrock.budget.used} of {bedrock.budget.cap}. Request shapes work for these models; the full run is safe to try.")
    return 0


def main(argv: list[str]) -> int:
    args = list(argv)
    flags = {"live": False, "yes": False, "smoke": False, "plan": False, "offline": False, "report": None, "answers": 1}
    try:
        while args:
            arg = args.pop(0)
            if arg in ("--live", "--yes", "--smoke", "--plan", "--offline"):
                flags[arg[2:]] = True
            elif arg == "--answers" and args:
                flags["answers"] = int(args.pop(0))
                if not 0 <= flags["answers"] <= len(ANSWER_SAMPLE):
                    raise ValueError(f"--answers must be between 0 and {len(ANSWER_SAMPLE)}")
            elif arg == "--report":
                flags["report"] = Path(args.pop(0)) if args and not args[0].startswith("--") else RESULTS_FILE
            elif arg == "--frozen":
                problems = ev.check_frozen_v3()
                print("question set untouched since it was frozen" if not problems else "CHANGED: " + "; ".join(problems))
                return 1 if problems else 0
            else:
                print(f"unknown or incomplete argument: {arg}")
                return 2
        if flags["report"]:
            if not flags["report"].is_file():
                print(f"no saved results at {flags['report']}; run the pipeline first")
                return 2
            print_report(json.loads(flags["report"].read_text(encoding="utf-8")))
            return 0
        data = Data()
        bd.load_env()
        config = bd.load_config(offline=not flags["live"])
        if flags["plan"] or (flags["live"] and not (flags["yes"] or flags["smoke"])):
            print_plan(make_plan(data, flags["answers"]), config.cap)
            if flags["live"] and not flags["plan"]:
                print("\nNothing was spent. Add --yes to run this for real (or --smoke first for about 60 tokens).")
                return 2
            return 0
        budget = bd.TokenBudget(config.cap)
        bedrock = bd.Bedrock(bd.make_client(config), config, budget)
        if flags["smoke"]:
            return smoke(bedrock)
        problems = ev.check_frozen_v3()
        if problems:
            print("WARNING: " + "; ".join(problems))
        print_plan(make_plan(data, flags["answers"]), config.cap)
        res = run_pipeline(bedrock, data, flags["answers"])
        save_results(res)
        print_report(res)
        print(f"\nResults saved to {RESULTS_FILE}. No prompts or credentials are stored in it.")
        return 0
    except (bd.ConfigError, bd.LiveCallError, ValueError) as err:
        print(f"error: {err}")
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
