# r07: Evaluation on a bigger, harder test set

> *Write the questions first, say what you expect, then see whether you were right.*

Equivalent in the Bedrock course: **b17 The evaluator loop**. A system you cannot score is a system you cannot improve, and a score without uncertainty or cost is a rumour.

## The problem

r05 ended with "the graph does not beat plain search", but the test could not show it either way: ten short documents, twelve questions written alongside them, plain search already at 0.97. A comparison needs a harder test, an honest floor, a way to say whether a gap is real, and the cost on the same page.

## What was added

| Piece | What it is |
|---|---|
| `corpus_large/` | 14 new original documents (guardrails, approval, sandboxing, retries, prompt caching, observability, evaluation, multi-agent, planning, checkpoints, queues, secrets, embeddings, reranking). With the 10 originals: 24 documents, 90 chunks. |
| `questions_v2.json` | 27 new questions. 9 are `keyword` style (they reuse the answer's words, as a control); 18 are `paraphrase` style. 13 are local (one gold document) and 14 are global (two to four). The original 12 are kept, so 39 in all. |
| Random baseline | A seeded random ranking of all chunks. The floor every method must beat. |
| Paired bootstrap | Resamples whole questions to give a 95% interval for "method minus BM25". If the interval includes 0, the data cannot tell them apart. |
| Frozen test set | `FROZEN.json` holds hashes of the new documents and questions, taken before the first run. A test fails if either changes, so the questions cannot be quietly tuned. |
| `PREDICTIONS.md` | Eight predictions written before the first run. The last section of the output scores them. |
| Cost model | Model calls and words for indexing and per question. Tokens are an estimate (1.3 per word) and no prices are assumed. |

## Run

```
python r07_evaluation/code.py --offline          # the full report
python r07_evaluation/code.py --misses bm25      # which questions a method fails
python r07_evaluation/code.py --frozen           # has the test set been touched?
```

## What was found (offline fakes, 24 documents, recursive 40-word chunks)

Coverage = share of the gold documents reached by the top 6 chunks.

| Method | Original 12 | keyword (9) | paraphrase (18) | All 39 | Prompt words |
|---|---|---|---|---|---|
| random | 0.03 | 0.13 | 0.12 | 0.09 | 180 |
| BM25 (plain search) | 0.97 | 1.00 | 0.85 | 0.92 | 177 |
| fake vector search | 0.97 | 1.00 | 0.62 | 0.82 | 169 |
| local | 0.82 | 0.96 | 0.52 | 0.72 | 953 |
| global | 0.69 | 0.22 | 0.25 | 0.38 | 788 |
| hybrid | 0.82 | 0.96 | 0.47 | 0.69 | 1,008 |
| mix | 0.86 | 0.96 | 0.56 | 0.75 | 1,005 |
| hybrid, re-ordered by keyword score | 0.94 | 1.00 | 0.76 | 0.87 | 996 |

On paraphrase questions the paired bootstrap says every graph mode is **clearly worse** than BM25 (for example hybrid minus BM25 is -0.38, 95% interval -0.59 to -0.15). Re-ordering narrows the gap to about -0.09 and never closes it. The graph modes also cost about 9 times the prompt words and one model call per question.

**Indexing cost.** 90 model calls for 24 documents, about 15,600 tokens in and 6,000 out (estimate). Plain search needs none. Editing one document re-extracts only its changed chunks (r06).

## The predictions, scored

| # | Prediction | Result |
|---|---|---|
| P1 | Random coverage at k=3 below 0.15 | **Held** (0.07) |
| P2 | BM25 at k=6: keyword >= 0.80, paraphrase <= 0.60 | **Wrong**: keyword 1.00, paraphrase 0.85 |
| P3 | Fake vector search no better than BM25 on paraphrase | **Held** (0.62 against 0.85) |
| P4 | No graph mode clearly above BM25 on paraphrase | **Held**, and more strongly: all four are clearly below |
| P5 | Graph modes cost 5 to 9 times BM25's prompt words | **Wrong**: 9.2 to 9.8 times |
| P6 | BM25 on the original 12 stays at 0.85 or more | **Held** (0.97, unchanged by 14 extra documents) |
| P7 | Re-ordered hybrid within 0.10 of BM25, not above it | **Held** (0.87 against 0.92) |
| P8 | Paraphrase intervals about 0.08 to 0.30 wide | **Wrong**: mean width 0.38 |

Five held and three were wrong. The wrong ones are the useful ones.

## What the wrong predictions say

- **P2: my "paraphrase" questions are not hard enough.** They share on average 58% of their words with the answers (the keyword ones 84%), and 6 of 18 share 70% or more. BM25 fails only 6 questions at k=6, and every failure is a multi-document question where 6 chunks cannot cover 3 or 4 gold documents. It answers every single-document paraphrase question. The questions are frozen, so this stays as a finding: the test set separates methods, but it is weaker on paraphrase than its label says. Better paraphrases should be written by someone who has not read the documents (a person, or a model that is only shown topics).
- **P8: 18 questions is too few.** Intervals about 0.38 wide mean only large gaps are detectable. Small differences between methods cannot be settled with this set.
- **P5** was only slightly off: the graph context is a little longer than I guessed.

## Two things not to over-read

1. **"mix, re-ordered" equals BM25 exactly (difference 0.00) by construction.** The mix pool contains BM25's own top 6 chunks and the re-orderer is BM25, so at k up to 6 it cannot do anything else. Its top 6 chunks match BM25's for 35 of 39 questions (the rest are score ties). A test pins this down so nobody mistakes it for a result.
2. **Everything here is lexical.** The keyword model, the extraction model and the "embedder" are fakes that only see shared words. This run shows the test set can separate methods; it does not show how a real model or real embeddings handle paraphrase. A real embedding model is the natural fix for the paraphrase gap that plain keyword search leaves, and the fake vector search cannot test that.

## What would justify a graph here

Nothing in this run does. The honest reading is: with fake models, on 24 short documents, graph retrieval costs about 9 times as much per question and scores lower. LightRAG's own README positions it for large collections and cross-document questions, so a 24-document corpus may simply be too small for a graph to pay off; I have not tested that. r08 is where a real model decides it, on a small, capped run.

## Exercises

1. Write five new paraphrase questions without looking at the documents, add them as `questions_v3.json` (do not edit v2), and see whether BM25 still scores above 0.8.
2. Raise `naive_pool` in the retriever and see how the `mix` rows move.
3. Run `--misses hybrid`. Which kinds of question does the graph lose that BM25 keeps?
