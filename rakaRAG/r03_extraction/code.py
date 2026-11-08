"""r03: Entity and relation extraction, with the answer checked independently.

Motto: ask the model for structure, then check every row yourself.

Each chunk goes to a model with a fixed prompt; the model answers with delimited rows; our
validator keeps good rows and rejects bad ones with a reason. Offline, the "model" is a fake
(a word list plus sentence co-occurrence), so the same prompt and validator will later run
against a real model in r08. Extraction costs one model call per chunk, so the chunking choice
from r02 now has a price.

Run:  python r03_extraction/code.py --offline
      python r03_extraction/code.py --show 04_hooks
      python r03_extraction/code.py --demo-faults
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import raka_chunking as ck  # noqa: E402
import raka_common as rc  # noqa: E402
import raka_extract as ex  # noqa: E402
from raka_fake_model import FAULTS, FakeExtractor  # noqa: E402

# (label, strategy, size in words, overlap in words)
CONFIGS = [
    ("whole document", "doc", 0, 0),
    ("recursive 40", "recursive", 40, 0),
    ("recursive 20", "recursive", 20, 0),
    ("fixed 40/10", "fixed", 40, 10),
]


def run_config(strategy: str, size: int, overlap: int, model=None):
    docs = rc.load_corpus()
    chunks = ck.chunk_corpus(docs, strategy, size, overlap)
    results = ex.extract_corpus(chunks, model or FakeExtractor())
    return chunks, results


def summarise(results: list[ex.Extraction]) -> dict:
    names = [e.key for r in results for e in r.entities]
    pairs = {frozenset((x.source.casefold(), x.target.casefold())) for r in results for x in r.relations}
    return {
        "calls": len(results),
        "prompt_words": sum(r.prompt_words for r in results),
        "output_words": sum(r.output_words for r in results),
        "entities": len(names),
        "unique_entities": len(set(names)),
        "relations": sum(len(r.relations) for r in results),
        "unique_pairs": len(pairs),
        "rejected": sum(len(r.rejected) for r in results),
        "incomplete": sum(1 for r in results if not r.complete),
    }


def print_report() -> None:
    print("\nOne model call per chunk. Words are a rough cost proxy (prompt = instructions + chunk text).")
    print(f"  {'chunking':<16}{'calls':>6}{'prompt w':>10}{'out w':>7}{'entities':>9}{'unique':>7}{'relations':>10}{'pairs':>6}{'rejected':>9}")
    for label, strategy, size, overlap in CONFIGS:
        _, results = run_config(strategy, size, overlap)
        s = summarise(results)
        print(f"  {label:<16}{s['calls']:>6}{s['prompt_words']:>10}{s['output_words']:>7}{s['entities']:>9}{s['unique_entities']:>7}{s['relations']:>10}{s['unique_pairs']:>6}{s['rejected']:>9}")
    print("\n'entities' counts every mention; 'unique' counts distinct names. The gap is what r04 must merge. "
          "'pairs' counts distinct related entity pairs across all chunks.")
    print("Smaller chunks mean more calls, and every call repeats the instructions, so cost grows faster than the text does.")


def show(doc_id: str, strategy: str, size: int) -> int:
    docs = {d.id: d for d in rc.load_corpus()}
    if doc_id not in docs:
        print(f"unknown document {doc_id!r}; choose one of: {', '.join(sorted(docs))}")
        return 2
    model = FakeExtractor()
    for chunk in ck.chunk_doc(docs[doc_id], strategy, size):
        result = ex.extract_chunk(chunk, model)
        print(f"\n[{chunk.id}] {chunk.words} words: {chunk.text}")
        for e in result.entities:
            print(f"  entity   {e.name} ({e.type}): {e.description[:70]}")
        for x in result.relations:
            print(f"  relation {x.source} -- {x.target} [{', '.join(x.keywords)}]")
        for row, why in result.rejected:
            print(f"  REJECTED ({why}): {row[:70]}")
    return 0


def demo_faults() -> int:
    docs = {d.id: d for d in rc.load_corpus()}
    chunk = ck.chunk_doc(docs["04_hooks"], "recursive", 30)[0]
    print(f"Chunk {chunk.id}: {chunk.text}\n")
    print(f"{'fault injected':<20}{'rows rejected':>14}  what the validator said")
    for fault in FAULTS:
        result = ex.extract_chunk(chunk, FakeExtractor(faults=(fault,)))
        said = "; ".join(sorted({why for _, why in result.rejected} | {n for n in result.notes if "recovered" in n or "completion" in n}))
        print(f"{fault:<20}{len(result.rejected):>14}  {said}")
    return 0


def main(argv: list[str]) -> int:
    args = list(argv)
    show_doc, strategy, size, demo = None, "recursive", 30, False
    try:
        while args:
            arg = args.pop(0)
            if arg == "--show" and args:
                show_doc = args.pop(0)
            elif arg == "--strategy" and args:
                strategy = args.pop(0)
            elif arg == "--size" and args:
                size = int(args.pop(0))
            elif arg == "--demo-faults":
                demo = True
            elif arg == "--offline":
                continue  # accepted for parity with the Bedrock course; r03 is always offline
            else:
                print(f"unknown argument: {arg}")
                return 2
        if demo:
            return demo_faults()
        if show_doc:
            return show(show_doc, strategy, size)
    except ValueError as err:
        print(f"error: {err}")
        return 2
    print("rakaRAG r03: entity and relation extraction with a fake model, offline")
    print_report()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
