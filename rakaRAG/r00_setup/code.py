"""r00: Setup and the test set.

Motto: no scoring without a question list that has known answers.

Loads the corpus and the question file, checks that every question points at real
documents, and prints what we have. Nothing calls a model or a network.

Run:  python r00_setup/code.py --offline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import raka_common as rc  # noqa: E402


def describe() -> tuple[dict, list[str]]:
    """Return (facts, problems). Problems empty means the test set is usable."""
    docs = rc.load_corpus()
    questions = rc.load_questions()
    problems = rc.validate_questions(questions, docs)
    if not docs:
        problems.append("the corpus folder has no .md files")
    words = sum(len(rc.tokenize(d.text)) for d in docs)
    facts = {
        "documents": len(docs),
        "indexed_words": words,
        "questions": len(questions),
        "local": sum(1 for q in questions if q["type"] == "local"),
        "global": sum(1 for q in questions if q["type"] == "global"),
    }
    return facts, problems


def main(argv: list[str]) -> int:
    facts, problems = describe()
    print("rakaRAG r00: setup check (offline, no model, no network)")
    for key, value in facts.items():
        print(f"  {key:<14}{value}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  - " + p)
        return 1
    print("OK: every question points at real documents.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
