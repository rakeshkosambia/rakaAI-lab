# r02: Chunking

> *Split the text where the meaning splits, and remember where each piece came from.*

Equivalent in the Bedrock course: the chunking step of a Knowledge Base (b10). Here we build it ourselves so we can see what each choice does.

## The problem

A whole document is a blunt unit. If one sentence answers the question, sending the whole document wastes prompt space and money, and buries the answer in noise. But cut carelessly and you slice a sentence in half, or lose the words around it that give it meaning.

## The idea

A **chunk** is a piece of a document plus the exact place it came from (character start and end). That span matters later: a citation must point at real text.

Three strategies, all counting size in words:

| Strategy | How it cuts | Good at | Weak at |
|---|---|---|---|
| `doc` | no cutting; the whole document is one chunk | nothing lost | large prompts |
| `fixed` | a window of `size` words that slides forward by `size - overlap` | simple, predictable sizes | cuts mid-sentence |
| `recursive` | split on the biggest natural boundary first (blank line, line, sentence, clause, word), then merge neighbours back up to `size` words | keeps sentences whole | sizes vary |

**Overlap** repeats a few words at the edge of neighbouring windows so a sentence cut in two still appears whole in one of them. It costs extra chunks. The rule from LightRAG's own chunker applies: overlap must be smaller than size, or the window would never move forward.

LightRAG's default is a fixed window (1,200 tokens, 100 overlap) and it also offers recursive and semantic splitters. We copy the ideas, not the code.

## Run

```
python r02_chunking/code.py --offline          # score six configurations, top 3 chunks
python r02_chunking/code.py --k 6              # same, top 6 chunks
python r02_chunking/code.py --show 04_hooks --strategy recursive --size 30
python r02_chunking/code.py --show 04_hooks --strategy fixed --size 30 --overlap 10
```

## What the scores mean

For each question we take the top k chunks and report:

| Column | Meaning |
|---|---|
| coverage | Fraction of the gold documents that at least one retrieved chunk comes from. |
| mrr | 1 / rank of the first chunk that comes from a gold document. |
| words | Average words those chunks put in the prompt. Lower is cheaper. |

## What you will see (current corpus, BM25)

| Chunking | Chunks | Prompt words at k=3 | Coverage at k=3 | Coverage at k=6 |
|---|---|---|---|---|
| whole document | 10 | 329 | 1.00 | 1.00 |
| fixed 40/10 | 40 | 112 | 0.97 | 0.97 |
| recursive 40 | 40 | 91 | 0.97 | 0.97 |
| recursive 20 | 80 | 41 | 0.97 | 1.00 |

Chunking cuts the prompt by roughly 3 to 8 times for a small loss of coverage, and the loss is on multi-document (`global`) questions: with only 3 chunks, two of them may come from the same document. Raising k brings most of it back. That is the real trade: smaller prompts against needing more chunks.

Do not over-read these numbers. Ten short documents is a toy corpus; r07 repeats this on something bigger and harder.

## Things to notice

- Run the two `--show` commands. The fixed window cuts in the middle of a sentence; the recursive splitter does not. The test `test_recursive_cuts_fewer_sentences_in_half_than_a_fixed_window` checks this.
- Every chunk is exactly `document[start:end]`. Tests enforce it for all strategies.
- No word is ever lost: every non-space character of a document lies inside some chunk.
- The `doc` row of r02 matches r01 exactly, which is a built-in check that the new code did not change the old results.

## Exercises

1. Add `fixed 80/20` to `CONFIGS` in `code.py` and compare it with `recursive 40`.
2. Make the recursive splitter prefer `"\n"` over `". "` for a document with bullet lists (change `SEPARATORS`) and see what breaks.
3. Why does overlap help `fixed` more than it would help `recursive`?
