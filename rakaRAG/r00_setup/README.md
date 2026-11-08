# r00: Setup and the test set

> *No scoring without a question list that has known answers.*

Equivalent in the Bedrock course: **b00 Setup**. There b00 checks your account; here r00 checks that our test set is sound, because everything later is judged against it.

## The problem

"The search feels better" is not a result. To compare keyword search, vector search and graph search we need the same fixed questions, each with the documents that really contain the answer written down in advance.

## The idea

A test set has three parts, all in this folder tree:

| Part | Where | What it holds |
|---|---|---|
| Corpus | `corpus/*.md` | 10 short original notes about agent harnesses. The file name (without `.md`) is the document id. |
| Questions | `questions.json` | 12 questions. Each has an id, a type, the gold documents that contain the answer, and phrases a good answer should contain (used from r05 on). |
| Types | `local` / `global` | A `local` question needs one document. A `global` question needs facts from two or more documents, the kind graph retrieval is meant to help with. |

`r00` loads them and refuses to pass if a question points at a document that does not exist, has no gold documents, has a duplicate id, or is marked `global` with fewer than two gold documents.

## Run

```
python r00_setup/code.py --offline
```

Expected: 10 documents, 12 questions (8 local, 4 global) and `OK`.

## Things to notice

- Gold documents are written down before any search runs. Never edit them after seeing a result to make a method look better.
- The set is small and the documents and questions share vocabulary. See "Known limits" in the main README.

## Exercises

1. Add a document and a question about it; run r00 and watch it pass.
2. Break a gold id on purpose and read the error.
