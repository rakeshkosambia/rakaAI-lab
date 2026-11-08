# r01: The two baselines

> *Before building something clever, measure something simple.*

Equivalent in the Bedrock course: the plain "naive" retrieval you would get before any Knowledge Base features (b10). Here we build both baselines ourselves.

## The problem

Graph RAG costs more to build (a model reads every chunk to extract entities). It is only worth it if it beats what a few lines of code can do. So we build the simple searches first and score them.

## The idea

**Keyword search (BM25).** Count the question's words in each document. A rare word counts more than a common one, and a long document is not rewarded just for being long.

**Vector-only ("naive") search.** Turn the question and each document into a vector and rank by closeness. In this chapter the vectors come from a fake hashing embedder: it only sees shared words, so it has no sense of meaning. That is deliberate and offline. A real embedding model arrives in r08.

**Scoring.** For each question we know the gold documents.

| Measure | Meaning |
|---|---|
| recall@k | Of the gold documents, what fraction appear in the top k results. |
| MRR | 1 divided by the rank of the first gold document; 1.00 means it is always first. |

Results are reported for all questions, for `local` ones and for `global` ones.

## Run

```
python r01_baselines/code.py --offline
python r01_baselines/code.py --query "What does a pre-tool hook do?"
python r01_baselines/code.py --k 5
```

## What you will see

On the current set BM25 scores 1.00 and the fake vector search scores 0.94 recall@3 (its misses are on global questions). Do not read this as "BM25 is great". The documents and the questions were written together and share words, so the test is easy. This chapter proves the scoring harness works; r07 adds paraphrased and multi-document questions that make the comparison informative.

## Things to notice

- Both methods return a full ranking, with ties broken by document id, so a run repeats exactly.
- A document with no matching word is never returned as a hit; the scorer places unmatched documents at the end.
- Global questions are where vector-only search slipped first. That is the gap graph retrieval is meant to close.

## Exercises

1. Change `k1` and `b` in `BM25Index` and see what moves. Then change them back.
2. Write a question whose answer uses different words from the document (a paraphrase) and see which method breaks first.
