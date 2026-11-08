# r08 predictions, written before any live run and before BM25 was run on `questions_v3.json` (2026-11-08)

Decisions behind this run (Rakesh, 2026-11-08): provider Bedrock; a cheap model for extraction, a stronger one for answers, a separate embedding model; hard cap of 20,000 tokens for the whole run; 15 paraphrase questions built from the concepts of the sparring-partner-ooda skill (blast radius, an assumption proving wrong, ten times the load, locked decisions, open items, a procedure executable without questions). The questions are in `questions_v3.json` and frozen by hash in `FROZEN.json`.

Setup: 24 documents, whole-document chunks (24 chunks, to fit the cap), 33 retrieval questions (the 15 new ones plus the 18 paraphrase questions from v2), top 3 and top 5 documents.

| # | Prediction | How it is checked |
|---|---|---|
| L1 | Real-embedding vector search reaches at least as many gold documents as BM25 on the 15 new questions at k=3 (mean coverage no lower). | Live results. |
| L2 | The 15 new questions are harder for BM25 than the v2 paraphrase questions were: BM25 coverage at k=3 on the new 15 is below its coverage on the 18 v2 paraphrase questions. (The v3 wording avoids the documents' vocabulary by design.) | Live results; BM25 needs no AWS. |
| L3 | No graph mode (local, global, hybrid) is clearly above real vector search on the new 15 at k=3: the 95% interval of the difference includes 0 or lies below it. | Paired bootstrap. |
| L4 | The cheap model returns usable keyword JSON for at least 13 of the 15 questions (the validator accepts it without the fallback). | Count of fallbacks. |
| L5 | At most 20% of extraction rows are rejected by the validator, and at least 20 of the 24 documents finish with the completion marker. | Extraction summary. |
| L6 | The whole run uses between 12,000 and 20,000 tokens and never exceeds the cap. | Budget log. |
| L7 | Every sampled answer cites only source ids it was given (the citation check passes), for at least 2 of the sampled answers. | Answer validation. |

What would change my mind: if real vector search is no better than BM25 on questions written this way (L1 wrong), the wording is not as far from the documents as intended and the test needs rewriting before any conclusion; if a graph mode is clearly above vector search (L3 wrong), I should first look for a flaw in my own test before believing it.

Known weaknesses, stated in advance:
- I wrote the documents and also chose the gold documents for these questions after reading them. The questions are phrased from the skill's concepts and not from the documents' text, which lowers shared vocabulary, but this is not a blind test by an independent author.
- 15 questions gives wide intervals; only large differences are detectable.
- Chunking by whole document and capping output keeps the run inside 20,000 tokens but is not how a production index would be built.
- Entities and relations are matched with keyword scoring and not embeddings (to save tokens), so the graph side is weaker than LightRAG's own design.
- The embedding and Converse request shapes were written from AWS documentation and tested only against an offline fake, so the first live call may need a small fix. `--live --smoke` is there to find that for about 60 tokens.
