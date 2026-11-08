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
| r07 | Evaluation on a bigger, harder test set; cost per document | planned |
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
```

Windows wrappers: `run.bat r01_baselines --offline` and `test.bat` (tests need `pip install -r requirements-dev.txt`).

## Known limits of the current test set

The 10 documents and 12 questions are small and were written together, so they share vocabulary. BM25 scores a perfect 1.00 on them. That means the current set cannot show a graph helping (r05 confirms it: the graph modes do not beat plain search here, and cost much more). Chapter r07 must add paraphrased questions (the right answer shares few words with the question), more documents, and questions that need facts from several documents. Until then, treat the numbers as a check that the scoring harness works, not as a result.

## Data rules

- Only our own notes and public text go into the corpus. No client data, no resumes, no credentials.
- Never commit `.env`. `.env.example` holds empty placeholders only.
- Text files are ASCII so Windows consoles print them safely.
