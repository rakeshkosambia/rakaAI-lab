# Predictions, written before the first r07 run (2026-11-08)

Written after the 14 new documents and the 27 questions in `questions_v2.json` were finished, and before any retrieval was run against them. The frozen hashes of both are in `FROZEN.json`. If a prediction is wrong, that is a finding, not a reason to edit the question set.

Setup: 24 documents (10 original + 14 new), recursive 40-word chunks, 12 original questions plus 27 new ones, `k` = 3 and 6. All models are the offline fakes: a lexical keyword model, a lexical extraction model and a hashing "embedder" that only sees shared words.

| # | Prediction | How it will be checked |
|---|---|---|
| P1 | A random chunk ranking reaches few gold documents (coverage at k=3 below 0.15). | Seeded random retriever. |
| P2 | BM25 stays strong on the 9 keyword-style questions (coverage at k=6 of 0.80 or more) and falls on the 18 paraphrase-style ones (0.60 or less). | Per-style coverage. |
| P3 | The fake vector search is no better than BM25 on paraphrase questions, because it only sees shared words. | Same table. |
| P4 | No graph mode beats BM25 on paraphrase questions: the 95% interval of (mode minus BM25) includes zero or lies below it. The fake extraction and keyword models are lexical, so the graph inherits BM25's blindness. | Paired bootstrap on per-question differences. |
| P5 | Graph modes again cost 5 to 9 times the prompt words of naive search, plus one model call per question. | Words and calls columns. |
| P6 | On the original 12 questions, BM25 coverage at k=3 drops a little from 0.97 now that there are 24 documents (more distractors), but stays at 0.85 or more. | Same questions, bigger corpus. |
| P7 | Re-ordering a graph pool by keyword score brings it to about BM25's level and no higher. | Rerank rows. |
| P8 | With 27 new questions, bootstrap intervals will be wide (roughly 0.10 to 0.20 across), so only large differences will be detectable. | Interval widths. |

What would change my mind: a graph mode whose interval lies clearly above zero on paraphrase questions would contradict P4, and I would then look for a flaw in my own test set before believing it.

Known weaknesses of this set, stated in advance: the same author wrote the documents and the questions; gold documents were chosen by reading the documents, not by an independent person; 27 questions is small; and nothing offline can judge paraphrase fairly because the fake models are lexical. r08 (a real model and real embeddings) is where that is tested.
