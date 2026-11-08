# r04: The graph store and entity merge

> *One node per thing, however many times the text mentions it.*

Equivalent in the Bedrock course: the graph a managed GraphRAG feature would build for you (b10). Here we build the store ourselves so the merge rules are visible.

## The problem

r03 produced 112 entity mentions that are really 21 different things. "Hook", "Hooks" and "a hook" must become one node, and a relation stated in three chunks must become one edge that remembers it was stated three times. Merge too little and the graph is a pile of duplicates. Merge too much and two different things become one.

## The idea

**Nodes and edges.** One node per entity, found by a normalised key (lower case, single spaces, no leading "the", last word un-pluralised). One undirected edge per pair of entities: A-B and B-A are the same edge.

**Identity is declared, never guessed.** "MCP" and "Model Context Protocol" are the same thing to us but not to a string comparison. We merge them only through an alias table we write ourselves (`KnowledgeGraph({"MCP": "Model Context Protocol"})`). Aliases are one hop only, no chains, and must be declared before any text is added, or earlier mentions would stay split.

**Everything derives from mentions.** A node stores one mention per chunk: its type, description and the exact spelling used. The visible facts are computed from them:

| Derived fact | Rule |
|---|---|
| type | the type most mentions agree on; a tie goes to the earliest chunk |
| name shown | the most common spelling |
| descriptions | collected in chunk order, exact duplicates dropped; "Entity NAME" if there are none |
| sources | the chunk ids it came from, ordered 2 before 10 |
| edge weight | the number of chunks that stated the relation |
| edge keywords | the union, in order |

Because each chunk contributes one mention, adding the same chunk twice changes nothing, and re-extracting a chunk replaces its old mention. Order does not matter either: the saved file is identical whatever order the chunks were added in. This is also what will make deleting a document in r06 a clean recompute.

**Summaries.** A node mentioned in 14 chunks collects up to 14 descriptions. `compact()` replaces a long pile with one summary from an injected summarizer. In a real run each summary is a model call, so the method returns how many were made. A new mention clears a stale summary, because it no longer covers everything.

**Saving.** `save()` writes a sorted, ASCII-only JSON file (to `index/`, which git ignores). `load()` refuses a wrong version, a node with no mentions, or an edge that points at a missing node.

## Run

```
python r04_graph/code.py --offline                  # graphs for four chunkings
python r04_graph/code.py --top 8                    # the most connected entities
python r04_graph/code.py --node Subagent            # one entity: sources, descriptions, links
python r04_graph/code.py --path "Hook" "Memory"     # fewest-hops path between two entities
python r04_graph/code.py --compact                  # how many summary calls, how many words saved
python r04_graph/code.py --save                     # write index/graph.json
```

## What you will see

| Chunking | Mentions | Nodes | Edges | Groups | Isolated |
|---|---|---|---|---|---|
| whole document | 54 | 21 | 26 | 4 | 2 |
| recursive 40 | 91 | 21 | 29 | 3 | 1 |
| recursive 20 | 112 | 21 | 25 | 5 | 3 |
| fixed 40/10 | 102 | 21 | 28 | 4 | 2 |

- **Nodes stay at 21** however the text is cut: the merge does its job.
- **Edges change with chunking.** An edge needs two terms in the same chunk. Whole documents find 26 edges only because the fake model keeps at most 6 entities per call; lifting that cap gives 29, the same as recursive 40 (checked by running it). Recursive 20 finds fewer, 25. Thirteen corpus sentences are longer than 20 words, so they get cut and separate their terms. That is the likely cause; I did not isolate it.
- **Most connected:** Subagent and Tool (degree 6). `--path "Hook" "Memory"` gives Hook, Agent Loop, Conversation, Memory: three hops, and the edge labels show how crude the fake model's relations are.
- **Compaction:** 12 summary calls shrink about 1,400 description words to about 480. The fake summarizer is not a real summary; this only shows the mechanism and the call count.

## Things to notice

- Every node and edge lists its source chunks. A citation or a deletion can always find the text.
- Mutation check: breaking the edge ordering or the type vote makes the tests fail, so they test the rules and not just the happy path.
- The graph is only as good as extraction. If r03 invents or misses an entity, merging cannot fix that.

## Exercises

1. Declare `{"S3": "Object Storage"}` and see what changes in `--node`.
2. Two names that you merge by alias are really different things. How would you find out? (Hint: the mention counts and descriptions on the merged node.)
3. Why store mentions per chunk rather than one running description string per node?
