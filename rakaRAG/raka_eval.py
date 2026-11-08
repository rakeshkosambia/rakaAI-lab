"""Evaluation helpers for r07: uncertainty, test-set checks, a random baseline and a cost model. Pure standard library."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import raka_chunking as ck
import raka_common as rc
import raka_extract as ex

ROOT = rc.ROOT
TOKENS_PER_WORD = 1.3  # a rough English rule of thumb; real tokenisers differ, so treat token figures as estimates


# ---------------------------------------------------------------- uncertainty
def paired_bootstrap(a: list[float], b: list[float], resamples: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Mean of (a - b) per question, with a 95% percentile interval from resampling whole questions.

    The two lists are per-question scores for two methods on the SAME questions, in the same order.
    If the interval includes 0, the data cannot tell the two methods apart.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty lists of equal length")
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return sum(diffs) / n, means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def clear_difference(low: float, high: float) -> str:
    """'higher' / 'lower' when the interval excludes 0, else 'unclear'."""
    if low > 0:
        return "higher"
    if high < 0:
        return "lower"
    return "unclear"


# ---------------------------------------------------------------- the test set itself
def lexical_overlap(question: str, gold_texts: list[str]) -> float:
    """Share of the question's content words that also appear in the gold documents (1.0 = every word shared)."""
    words = set(rc.tokenize(question))
    if not words:
        return 0.0
    gold = set()
    for text in gold_texts:
        gold.update(rc.tokenize(text))
    return len(words & gold) / len(words)


def _normalised_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def frozen_hashes() -> dict:
    """The current hashes of the r07 test set, computed the same way as when it was frozen."""
    corpus = hashlib.sha256()
    files = sorted(rc.CORPUS_LARGE_DIR.glob("*.md"))
    for path in files:
        corpus.update(path.name.encode("ascii") + b"\n" + _normalised_bytes(path) + b"\n")
    return {
        "questions_v2_sha256": hashlib.sha256(_normalised_bytes(rc.QUESTIONS_V2_FILE)).hexdigest(),
        "corpus_large_sha256": corpus.hexdigest(),
        "corpus_large_files": len(files),
    }


def check_frozen(frozen_file: Path | str | None = None) -> list[str]:
    """Problems if the test set no longer matches the hashes recorded before the first run (empty = untouched)."""
    path = Path(frozen_file or ROOT / "r07_evaluation" / "FROZEN.json")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    now = frozen_hashes()
    return [f"{key} changed since it was frozen" for key in now if now[key] != recorded.get(key)]


# ---------------------------------------------------------------- a random baseline
class RandomRetriever:
    """Ranks all chunks in a seeded random order per question. The floor every method must beat."""

    def __init__(self, chunks: list[ck.Chunk], seed: int = 0):
        self.chunks, self.seed = list(chunks), seed

    def ranking(self, question: str) -> list[ck.Chunk]:
        rng = random.Random(int(hashlib.sha256(f"{self.seed}|{question}".encode("utf-8")).hexdigest()[:12], 16))
        order = list(self.chunks)
        rng.shuffle(order)
        return order


# ---------------------------------------------------------------- cost
def estimate_tokens(words: float) -> int:
    return math.ceil(words * TOKENS_PER_WORD)


def indexing_cost(index) -> dict:
    """What building this index costs: one model call per distinct chunk text, counted from what is stored.

    Words are the prompt (instructions + chunk) and the answer. Tokens are an estimate from words.
    """
    seen, prompt_words, output_words = set(), 0, 0
    for chunk in sorted(index.chunks.values(), key=lambda c: (c.doc_id, c.index)):
        key = index._key(chunk)
        if key in seen:
            continue
        seen.add(key)
        prompt_words += len(ex.build_prompt(chunk.text, index.params["max_rows"]).split())
        output_words += len(index.cache[key].split())
    return {"calls": len(seen), "prompt_words": prompt_words, "output_words": output_words,
            "tokens_in": estimate_tokens(prompt_words), "tokens_out": estimate_tokens(output_words),
            "documents": len(index.docs), "chunks": len(index.chunks)}


def query_cost(mode: str, keyword_calls: float, context_words: float, chunk_words: float, keyword_prompt_words: float) -> dict:
    """Per-question cost of a mode: extra model calls and the words that go into the answer prompt."""
    return {"mode": mode, "extra_calls": keyword_calls, "extra_prompt_words": keyword_prompt_words * keyword_calls,
            "answer_prompt_words": chunk_words + context_words}
