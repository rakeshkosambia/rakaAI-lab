# rakaRAG

A small graph-RAG built from scratch to learn the idea behind [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (MIT), then compared against plain search on our own material. It follows the same pattern as `learn-aws-bedrock/`: one folder per chapter, each with a `README.md` (the lesson) and a runnable `code.py`, offline first, live later.

This is original code. LightRAG is the inspiration; none of its code is copied. See `NOTICE.md` and `PLAN.md`.

## The question

Does graph plus vector retrieval beat plain keyword search and vector-only search on our own questions, and is the extra indexing cost worth it?

## Chapters

| Ch | Builds | Status |
|---|---|---|
| r00 | Setup, test corpus, question list with known answers | built, tested offline |
| r01 | Baselines: keyword (BM25) and vector-only search, scored | built, tested offline |
| r02 | Chunking: whole, fixed window, recursive; every chunk keeps its source span | built, tested offline |
| r03 | Entity and relation extraction: prompt, fake model, independent validator | built, tested offline |
| r04 | Graph store and entity merge: normalised keys, declared aliases, per-chunk mentions, summaries, save/load | built, tested offline |
| r05 | Query modes: naive, local, global, hybrid, mix; keyword validation, fallback, optional rerank; retrieval scored | built, tested offline |
| r06 | Incremental add, update, delete; extraction cache; rollback on failure; consistency check; save and load | built, tested offline |
| r07 | Evaluation: 24 documents, 39 questions, random floor, bootstrap intervals, cost, frozen test set, predictions scored | built, tested offline |
| r08 | Live run with a real model, small corpus, spend cap | planned |

## Quick start

Offline needs only Python 3.10+ and no packages.

```
python r00_setup/code.py --offline
python r01_baselines/code.py --offline
python r02_chunking/code.py --offline
python r03_extraction/code.py --offline
python r04_graph/code.py --offline
python r05_query_modes/code.py --offline
python r06_incremental/code.py --offline
python r07_evaluation/code.py --offline
```

Windows wrappers: `run.bat r01_baselines --offline` and `test.bat` (tests need `pip install -r requirements-dev.txt`).

## What we know so far, and what we do not

- **Chapters r00 to r05** use 10 short documents and 12 questions written together; BM25 scores about 1.00 on them, so they only prove the harness works.
- **r07** adds 14 documents and 27 questions (24 documents, 39 questions in all), a random floor, 95% bootstrap intervals and cost. Result: with the offline fake models, every graph mode scores clearly lower than plain keyword search on the paraphrase questions and costs about 9 times the prompt words plus a model call per question. Five of eight pre-registered predictions held; three were wrong, and one wrong one shows the "paraphrase" questions are less hard than labelled (they share 58% of their words with the answers).
- **Not yet known:** how a real model and real embeddings behave. Everything offline is lexical, so it cannot test paraphrase fairly. That is r08, which needs a model, a spend cap and a better paraphrase set written blind to the documents.

## Data rules

- Only our own notes and public text go into the corpus. No client data, no resumes, no credentials.
- Never commit `.env`. `.env.example` holds empty placeholders only.
- Text files are ASCII so Windows consoles print them safely.
