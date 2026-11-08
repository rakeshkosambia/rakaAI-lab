"""Shared helpers for the rakaRAG chapters: corpus loading, tokenising, a fake embedder, scoring.

Pure standard library. Nothing here calls a network or a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
QUESTIONS_FILE = ROOT / "questions.json"

QUESTION_TYPES = ("local", "global")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or so that the their then "
    "there these this to was when which while with without can may not more than such each if does "
    "do how what why who whose".split()
)

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str


def _stem(word: str) -> str:
    """A very light stemmer: enough to match tool/tools and policy/policies, nothing clever."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-case words, stop words removed, light stemming. Same function for documents and queries."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def load_corpus(corpus_dir: Path | str | None = None) -> list[Doc]:
    """Every *.md file in the corpus folder, sorted by file name. The id is the file name without .md."""
    folder = Path(corpus_dir) if corpus_dir else CORPUS_DIR
    docs = []
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
        docs.append(Doc(path.stem, title, text))
    return docs


def load_questions(path: Path | str | None = None) -> list[dict]:
    return json.loads(Path(path or QUESTIONS_FILE).read_text(encoding="utf-8"))


def validate_questions(questions: list[dict], docs: list[Doc]) -> list[str]:
    """Return a list of problems (empty means the question file is usable)."""
    problems: list[str] = []
    doc_ids = {d.id for d in docs}
    seen: set[str] = set()
    for q in questions:
        qid = q.get("id", "<no id>")
        for field in ("id", "question", "type", "gold_docs", "expect"):
            if field not in q:
                problems.append(f"{qid}: missing field {field}")
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        if q.get("type") not in QUESTION_TYPES:
            problems.append(f"{qid}: type must be one of {QUESTION_TYPES}")
        gold = q.get("gold_docs") or []
        if not gold:
            problems.append(f"{qid}: gold_docs is empty")
        for g in gold:
            if g not in doc_ids:
                problems.append(f"{qid}: gold doc {g} is not in the corpus")
        if q.get("type") == "global" and len(gold) < 2:
            problems.append(f"{qid}: a global question needs at least 2 gold docs")
    return problems


# ---------------------------------------------------------------- fake embedder
def hash_embed(text: str, dim: int = 512) -> list[float]:
    """A stand-in embedder: signed hashing of word tokens into a fixed-size unit vector.

    It has NO meaning-awareness (it only sees shared words), so it is a weak baseline on purpose.
    A real embedding model arrives in chapter r08.
    """
    vec = [0.0] * dim
    for token in tokenize(text):
        h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
        vec[h % dim] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # both vectors are unit length


# ---------------------------------------------------------------- scoring
def recall_at_k(ranked_ids: list[str], gold: list[str], k: int) -> float:
    """Fraction of the gold documents found in the top k."""
    return len(set(ranked_ids[:k]) & set(gold)) / len(gold) if gold else 0.0


def reciprocal_rank(ranked_ids: list[str], gold: list[str]) -> float:
    """1 / rank of the first gold document, or 0 when none is found."""
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold:
            return 1.0 / position
    return 0.0


def evaluate(search, questions: list[dict], k: int = 3) -> dict:
    """search(question_text) -> list of doc ids, best first (the full ranking).

    Returns per-question rows plus averages for all, local and global questions.
    """
    rows = []
    for q in questions:
        ranked = list(search(q["question"]))
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "gold": q["gold_docs"],
                "top": ranked[:k],
                "recall": recall_at_k(ranked, q["gold_docs"], k),
                "rr": reciprocal_rank(ranked, q["gold_docs"]),
            }
        )

    def mean(items, key):
        return sum(r[key] for r in items) / len(items) if items else 0.0

    summary = {}
    for label, subset in (
        ("all", rows),
        ("local", [r for r in rows if r["type"] == "local"]),
        ("global", [r for r in rows if r["type"] == "global"]),
    ):
        summary[label] = {"n": len(subset), "recall": mean(subset, "recall"), "mrr": mean(subset, "rr")}
    return {"k": k, "rows": rows, "summary": summary}


def evaluate_chunk_retrieval(search, questions: list[dict], k: int = 3, extra_words=None) -> dict:
    """Score a retriever that returns CHUNKS (anything with .doc_id and .text), best first.

    Takes the top k chunks and reports, per group (all / local / global):
      coverage  fraction of the gold documents that at least one of the k chunks comes from
      mrr       1 / rank of the first chunk that comes from a gold document
      words     average number of words those k chunks would put in the prompt (a cost proxy), plus
                extra_words(question) if given (for example graph context sent next to the chunks)
    """
    rows = []
    for q in questions:
        chunks = list(search(q["question"]))[:k]
        gold = q["gold_docs"]
        covered = {c.doc_id for c in chunks}
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "coverage": len(covered & set(gold)) / len(gold) if gold else 0.0,
                "rr": next((1.0 / pos for pos, c in enumerate(chunks, 1) if c.doc_id in gold), 0.0),
                "words": sum(len(c.text.split()) for c in chunks) + (extra_words(q["question"]) if extra_words else 0),
            }
        )

    def mean(items, key):
        return sum(r[key] for r in items) / len(items) if items else 0.0

    summary = {}
    for label, subset in (
        ("all", rows),
        ("local", [r for r in rows if r["type"] == "local"]),
        ("global", [r for r in rows if r["type"] == "global"]),
    ):
        summary[label] = {"n": len(subset), "coverage": mean(subset, "coverage"), "mrr": mean(subset, "rr"), "words": mean(subset, "words")}
    return {"k": k, "rows": rows, "summary": summary}
