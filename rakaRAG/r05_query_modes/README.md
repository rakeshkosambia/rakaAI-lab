# r05: Query modes

> *A graph only earns its cost if it finds what plain search misses.*

Equivalent in the Bedrock course: the retrieval step of a Knowledge Base (b10), plus the "is it worth it" check that b17 (the evaluator loop) will formalise.

## The problem

Plain chunk search matches a question's words to a chunk's words. A question like "which mechanisms keep the context window small?" needs facts from three documents that may not look like the question. Graph RAG tries to follow links between named things instead. We now have the graph (r04); does it help?

## The idea

Five modes, from LightRAG's design, rebuilt here:

| Mode | How it finds chunks | Model calls |
|---|---|---|
| `naive` | plain keyword search over chunks (the baseline) | 0 |
| `local` | keywords naming specific things -> matching entities -> their relations -> the chunks that mention them | 1 |
| `global` | keywords about themes -> matching relations -> their entities -> the chunks that state them | 1 |
| `hybrid` | local and global together | 1 |
| `mix` | hybrid plus the naive chunks | 1 |

The one call is keyword extraction. A model reads the question and returns two lists as JSON: **low-level** keywords (specific named things) and **high-level** keywords (themes). Our validator checks that answer itself: it takes the first `{...}` block, requires exactly the two keys, lists of short strings, drops duplicates and placeholders, caps each list at 8. If the answer is unusable we fall back to the question's own words and the result says `fallback=True`. We never ask the model to behave and trust it.

**How chunks are ranked in the graph modes.** Each selected entity and relation "votes" for the chunks it came from. A chunk with more votes ranks higher; ties go to keyword score. `rerank=True` instead orders the same pool by keyword score to the question.

Nothing here writes an answer. Answer generation needs a real model (r08). This chapter measures **retrieval**: do the top chunks reach the gold documents, and at what cost in prompt words and model calls?

## Run

```
python r05_query_modes/code.py --offline         # all five modes, top 3 and top 6 chunks, both orderings
python r05_query_modes/code.py --rerank          # only the re-ordered tables
python r05_query_modes/code.py --ask "Which mechanisms keep the context window small?" --mode hybrid
python r05_query_modes/code.py --demo-faults     # break the keyword model five ways
```

## What you will see (current corpus, recursive 40 chunks, 12 questions)

Coverage = share of the gold documents reached by the top chunks. Words = prompt size (chunks plus entity and relation text).

| Mode | Cov at k=3 | Cov at k=6 | Words at k=3 | Cov at k=3 if re-ordered |
|---|---|---|---|---|
| naive | 0.97 | 0.97 | 91 | 0.97 |
| local | 0.53 | 0.88 | 822 | 0.94 |
| global | 0.62 | 0.69 | 484 | 0.64 |
| hybrid | 0.74 | 0.88 | 828 | 0.97 |
| mix | 0.82 | 0.88 | 828 | 0.97 |

**The honest reading: the graph does not beat plain search here.**

1. **Cost.** Graph modes send about 5 to 9 times as many prompt words (entity descriptions are long and unsummarised) and need a model call per question. Naive needs none.
2. **Ordering, not finding.** In `local`, `hybrid` and `mix` the candidate pool contains every gold document (coverage 1.00), but the pool is huge (about 21 of 40 chunks) and the vote order buries the right chunks. Re-ordering the same pool by keyword score lifts hybrid from 0.74 to 0.97, which only equals naive. This is why LightRAG recommends a reranker.
3. **Global misses.** The `global` pool itself misses about 28% of the gold documents. The relation text comes from the fake extraction model, whose labels are crude, so theme matching has little to work with.
4. **The modes do behave as designed.** At k=6, `global` beats `local` on global questions (0.83 against 0.62), and `local` beats `global` on local questions (1.00 against 0.62). The intent is visible; the benefit is not, yet.

**Why not conclude "graph RAG is not worth it"?** Because this test cannot show it. Plain search already scores about 0.97 to 1.00 on documents and questions that were written together, so there is almost nothing for a graph to add. The extraction and keyword models are fakes. r07 needs paraphrased questions, a bigger corpus and a real model before any verdict.

## The fault demo

`--demo-faults` breaks the keyword model five ways (not JSON, a missing key, a wrong type, an extra key, an empty answer). Each is refused and the fallback runs. On the demo question the fallback happened to return a better third chunk than the fake keywords did, which tells you how little the fake keyword model adds.

## Things to notice

- The `naive` row equals r02's "recursive 40 / BM25" row exactly, a built-in check that nothing drifted.
- Hybrid is exactly the union of local and global; mix is exactly hybrid plus naive. Tests enforce both.
- Source chunk ids that the graph names but the store no longer holds are counted (`missing_sources`), not silently ignored. r06 (delete) will need that.

## Exercises

1. Run `--ask ... --mode local` and `--mode global` for the same question and compare the entities and relations each selects.
2. Shrink the pool: pass `top_entities=2, top_edges=3` to `Retriever`. Does coverage fall faster than words?
3. Call `graph.compact(fake_summarize)` from r04 before retrieval and see how the word cost changes.
