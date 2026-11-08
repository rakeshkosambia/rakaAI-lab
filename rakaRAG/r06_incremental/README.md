# r06: Incremental add, update and delete

> *A change to one document should cost one document, and leave nothing stale behind.*

Equivalent in the Bedrock course: keeping a Knowledge Base in step with its source documents (b10), and the "task state must survive a crash" idea from b11.

## The problem

Documents change. If any edit forces a full re-index you pay the model for every chunk again, and if a delete leaves entities, relations or chunk references behind, answers quietly cite text that no longer exists. Graph indexes make this hard: one entity is built from many documents, so deleting one document must not delete an entity another document still supports.

## The idea

The index keeps four things: **documents**, **chunks**, an **extraction cache** (the model's raw answer for each chunk), and the **graph**.

Because r04's graph stores one mention per chunk and derives everything from them, the operations are small:

| Operation | What it does | Model calls |
|---|---|---|
| `add_document` | chunk, extract (cache first), merge into the graph | one per chunk not in the cache |
| `delete_document` | remove the document's chunks; remove their mentions from the graph; drop nodes and edges left with no mentions | 0 |
| `update_document` | delete then add; unchanged chunks come from the cache | only changed text |

LightRAG's delete works the same way in outline: find every entity and relation that cites the document's chunks, delete those with no remaining sources, and rebuild the rest from cached extraction instead of asking the model again. Ours is simpler because "rebuild" is just "recompute from the remaining mentions".

**The cache key** is a hash of the prompt template, the row limit and the chunk text. So an answer is reused only when the model would be asked the same question again. Change the prompt and old answers stop matching. A deleted document's answers stay in the cache by default, so putting it back costs nothing; `drop_cache` removes them, but never an answer another document still needs.

**Failures leave nothing behind.** If the model fails halfway through a document, everything added so far is removed and the index is exactly as it was. Answers that did arrive stay in the cache, so the retry only pays for the rest.

**The doctor check, `verify()`.** After any operation it can prove the index is consistent: every chunk belongs to a document, every graph source exists, every edge has both ends, and the graph equals a fresh rebuild from the cache. It reports each problem in words. `save()` refuses to write an index that fails it, and `load()` refuses to open one.

**Saving.** Five JSON files, each written to a temporary file and swapped in, so a crash never leaves a half-written file. A crash between files could still leave a mixed set, which `load()` catches through `verify()`. The files are sorted and ASCII, so the same index saves byte-for-byte the same whatever order the documents were added in.

## Run

```
python r06_incremental/code.py --offline         # delete every document in turn, compare with a scratch build
python r06_incremental/code.py --delete 04_hooks
python r06_incremental/code.py --demo-update     # edit one sentence: how much is re-extracted?
python r06_incremental/code.py --demo-fail       # the model fails halfway through an add
python r06_incremental/code.py --save            # save, load back, verify
```

## What you will see

Full build: 10 documents, 40 chunks, 21 nodes, 29 edges, **40 model calls**.

Deleting any one of the 10 documents costs **0** model calls and produces exactly the same index as one that never saw the document (checked for all 10). Examples:

| Deleted | Nodes gone | Nodes kept (still mentioned elsewhere) | Edges gone |
|---|---|---|---|
| 04_hooks | 1 | 5 | 3 |
| 10_rag | 3 | 2 | 2 |
| 07_memory | 2 | 4 | 4 |
| 02_tools | 0 | 4 | 0 |

- **Update.** Appending one sentence to `04_hooks` re-extracts **1** of its 5 chunks (4 come from the cache), and the result equals a scratch build of the edited corpus. Updating again with the same text costs 0.
- **Failure.** The model fails on call 7, halfway through the second document. Two chunks had already been added; they are rolled back and the graph is unchanged. The retry makes 2 calls and takes 2 chunks from the cache.

## How we know it is right

- Every single-document delete equals a scratch build of what remains.
- Six random histories of 50 steps each (add, update, delete, sometimes dropping the cache) run `verify()` after every step and end equal to a scratch build of what remains.
- Three deliberate breaks each make a test fail: a delete that leaves chunks in the store, a delete that keeps edges with no mentions, and an add that does not roll back.
- Damage is detected: tests remove a chunk, a document, a node or the whole cache, and tamper with a description, and each is reported by `verify()`.

## Limits

- One process, no locking. LightRAG guards deletion with a pipeline lock because several jobs can run at once; we do not.
- A node that survives a delete loses its summary (it no longer matches). Re-run `compact()` to rebuild it, which costs a model call per summary.
- Chunk ids contain a position (`doc#2`), so editing the start of a document renumbers its chunks. That is why update is delete plus add, with the cache keyed by text and not by id.

## Exercises

1. Delete a document with `drop_cache=True`, then add it back. What does it cost?
2. Make two documents share a paragraph. Delete one with `drop_cache=True` and check the other still verifies.
3. Corrupt `index/rakarag/graph.json` by hand and try to load it.
