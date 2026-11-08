# rakaRAG as a showcase: what can honestly be said

Written 2026-11-08. Update the "live result" section after the Bedrock run. Every claim below points at something a reader can check in this repo.

## One line

**rakaRAG** is a graph-augmented retrieval pipeline built from scratch in plain Python to study the design of LightRAG, evaluated with pre-registered predictions, bootstrap uncertainty and a hard token cap on Amazon Bedrock.

## What is true today (and where to check it)

| Claim | Evidence |
|---|---|
| Nine chapters, r00 to r08, each with a lesson README and runnable code | `r00_setup/` to `r08_live/` |
| 346 offline tests, all passing | `test.bat` (needs only `pip install -r requirements-dev.txt`) |
| Chunking, extraction, a merged knowledge graph, five query modes, and incremental add, update and delete | `raka_chunking.py`, `raka_extract.py`, `raka_graph.py`, `raka_retrieve.py`, `raka_store.py` |
| The model's output is never trusted: extraction rows, keyword JSON and answer citations are each re-checked by code | `parse_records`, `parse_keywords`, `validate_answer`, with fault-injection demos (`--demo-faults`) |
| Deleting a document gives exactly the index you would get by never adding it, verified for every document and for random histories | `tests/test_store.py` |
| A hard token cap whose guard, stage pre-check and accounting were each deliberately broken to prove the tests notice | `raka_bedrock.py`, `tests/test_live.py` |
| Predictions were written before running, the question set is frozen by hash, and results include the predictions that were wrong | `r07_evaluation/PREDICTIONS.md`, `FROZEN.json`; `r08_live/PREDICTIONS.md`, `FROZEN.json` |
| The limitations are written down next to the results, not hidden | every chapter README |

## Results to date (offline, with fake models)

From r07, on 24 documents and 39 questions:

- With the fake models, every graph retrieval mode scored **clearly lower** than plain keyword search on the paraphrase questions (for example hybrid minus BM25 = -0.38, 95% interval -0.59 to -0.15), and cost about 9 times as many prompt words plus one model call per question.
- Five of eight pre-registered predictions held. Three were wrong, and the wrong ones were the informative ones: the "paraphrase" questions were less hard than labelled, 18 questions gave wide intervals, and the cost ratio was slightly higher than guessed.

Treat this as a result about **the test and the fakes**, not about graph RAG. Everything offline is lexical, so it cannot judge paraphrase fairly.

## Live result (fill in after the Bedrock run)

| Item | Value |
|---|---|
| Date and Region | _pending_ |
| Models (extraction / answers / embeddings) | _pending_ |
| Tokens used of the 20,000 cap | _pending_ |
| Real embeddings vs BM25 on the 15 new questions (k=3) | _pending_ |
| Best graph mode vs real vector search (95% interval) | _pending_ |
| Predictions L1 to L7 | _pending_ |

Do not publish anything that implies these are known until they are filled in from `index/r08_results.json`.

## What not to claim

- Any live Bedrock result, until the table above is filled in.
- That graph RAG is better or worse than plain search in general. At most: on this small corpus, with these questions and these models, here is what happened and with what uncertainty.
- That it is production-ready, or a benchmark against LightRAG itself. It is an independent rebuild of the idea at small scale; no LightRAG code is included.
- That the test set is independent. The same author wrote the documents and chose the gold documents. The questions in `questions_v3.json` were derived from the sparring-partner-ooda concepts and share only 29% of their words with the answers, which helps, but it is not a blind test.

## Wording you could use today (accurate as of now)

**Resume or profile bullets** (pick what fits):

- Built a graph-augmented retrieval (RAG) pipeline from scratch in Python as a study of LightRAG: chunking, entity and relation extraction with independent output validation, graph merge, five query modes and incremental updates, covered by 346 offline tests.
- Designed the evaluation before the results: frozen test set by hash, written predictions, random baseline and paired-bootstrap intervals; reported the predictions that turned out wrong.
- Added a hard token-budget guard for Amazon Bedrock calls and verified it by deliberately breaking it; staged the pipeline so the cheapest, most informative steps run first.

**A short post** (swap in the live result when you have it):

> I rebuilt the idea behind LightRAG from scratch to find out whether graph retrieval is worth its cost. Before running anything I wrote down eight predictions and froze the test questions by hash. Three predictions were wrong, and that taught me more than the five that held. With offline stand-ins the graph modes lost to plain keyword search at about 9x the prompt cost; next I am testing it with real Bedrock models under a 20,000-token cap. Code, tests and the honest write-up: github.com/rakeshkosambia/rakaAI-lab

Only post the last sentence about Bedrock once the run has happened, or say "next".

## What would make it stronger, in order of value

1. **Do the live run** (`r08_live/README.md`) and publish the result, including if it is negative. A negative result with intervals is more credible than a win without them.
2. **An independent question set.** Have another person, or a model shown only topic names, write 30 or more questions, and use a real public corpus instead of notes written alongside the questions.
3. **Compare against LightRAG itself** on the same corpus and questions, to turn "my rebuild" into "my rebuild against the reference".
4. **Polish for readers:** a licence for the original folders (the repo README says none is chosen yet), a CI workflow that runs the tests, a short architecture diagram and a terminal recording of r07 and r08.
5. **Close the known deviations:** token-based chunking, embedding entities and relations, and a real reranker.
6. **A short write-up** of the method (pre-registration, bootstrap, cost cap) that does not depend on the result.

## Repository hygiene before you share it widely

The repo is public. Tests check for secrets, emails and client paths; `.env` is git-ignored and `.env.example` holds placeholders only. Before linking it from a profile: re-read the READMEs once as a stranger would, and confirm no AWS account id, SSO start URL or personal data appears anywhere (the hygiene test looks for the common patterns, not every possible one).
