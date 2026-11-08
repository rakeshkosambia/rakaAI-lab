# rakaRAG -- plan (draft, 2026-11-08)

A small graph-RAG built from scratch to learn the idea behind [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG), then compared against plain search on our own material. This is a study project: original code, LightRAG credited as the inspiration, none of its code copied.

## What was checked in the LightRAG repo (read 2026-11-08, via the GitHub API)

| Fact | Value |
|---|---|
| Licence | MIT |
| Stars | 40,603 (the card said 37K) |
| Language | Python; last push 2026-11-05 |
| Core idea | Index documents into BOTH a knowledge graph (entities and relations) and vector embeddings, then retrieve from both |
| Query modes | `local` (specific entities), `global` (themes and relations), `hybrid` (local+global), `naive` (plain chunk vector search), `mix` (all three, the default) |
| LLM roles | Extraction (every chunk, cheap non-thinking model), Query (stronger), Keyword (fast, non-thinking), VLM (images) |
| Storage | 4 kinds: KV (cache, chunks, extractions), vector, graph, document status. Defaults are in-memory with local file persistence, "not for production"; PostgreSQL recommended for production |
| Cost notes | Entity extraction runs on every chunk; embedding model must be fixed before indexing (changing it means re-embedding everything) |
| Security note | The server binds to 0.0.0.0 by default and is open without auth configured |
| Prompts | `lightrag/prompt.py` (900 lines): entity-extraction prompt with tuple delimiters, optional JSON-output mode, description summarising, answer prompts for graph and naive modes, a keyword-extraction prompt |

Not yet studied in depth: `operate.py` (the indexing and query logic), the chunkers, deletion and incremental update, and the evaluation folder. Those are the reading for chapters r02 to r06.

## The question rakaRAG answers

Does graph + vector retrieval beat plain keyword search and vector-only search on our own questions, and is the extra indexing cost worth it?

## Chapters (offline first, then live, like learn-aws-bedrock)

| Ch | Builds | Tested offline with |
|---|---|---|
| r00 | Setup, test corpus, a question list with known answers | A small corpus in the repo |
| r01 | Baselines: keyword (BM25-style) search and vector-only search ("naive") | A fake hashing embedder, no network |
| r02 | Chunking: fixed-length and heading/paragraph-aware | Unit tests on sample markdown |
| r03 | Extraction: ask a model for entities and relations per chunk, validate the shape independently | A fake model returning canned JSON (rule: never trust the model's own format compliance) |
| r04 | Graph store and merge: same entity from many chunks becomes one node, descriptions merged | Unit tests |
| r05 | Query modes: local, global, hybrid, mix | Fixed graph, expected retrieved sets |
| r06 | Incremental add and delete of a document | Tests that deletion removes only what it owns |
| r07 | Evaluation: score all modes against the question list, and record cost per document | Report file |
| r08 | Live run with a real model on the user's own account, small corpus | Manual, cost-capped |

## Test corpus (own and public data only)

- Own docs: documentation, notes and memory files. No client data, no resumes, no credentials, no `.env`.
- Optional second corpus: about 500 public job postings with any client identifiers removed, for the question "which skills co-occur with a given role?".
- Each question has a written expected answer so scoring is checkable, not a feeling.

## Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Folder `rakaAI-lab/rakaRAG/`, original code, credit LightRAG in a NOTICE | proposed |
| 2 | Pure Python, minimal dependencies (standard library, plus numpy only if needed); no database server | proposed |
| 3 | Model and embedder are injected functions: fake in tests, real in r08 (Bedrock or Anthropic, user's choice) | proposed |
| 4 | Hosting: local on the PC only; no server, no open port | proposed |
| 5 | Success rule: graph modes must beat the baselines on the question list by a stated margin, or the result is recorded as "not worth it" | proposed |
| 6 | No client data of any kind, ever; secrets only in a git-ignored `.env`, with `.env.example` | proposed |

## Open items

| # | Item | Risk |
|---|---|---|
| 1 | Chunkers read for r02 (done 2026-11-08). `operate.py` extraction and validation read for r03 (done 2026-11-08). Node merge read for r04 (done 2026-11-08). Still to read before r05: edge merge details and the query functions | Low |
| 2 | Choose the live model and embedder for r08 and set a spend cap | Medium |
| 3 | Pick a licence for the lab's original folders (README says none chosen yet) | Low |
| 4 | Whether to link rakaRAG from b10 (Knowledge and RAG) in learn-aws-bedrock | Low |
