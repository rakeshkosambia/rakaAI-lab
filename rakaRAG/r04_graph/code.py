"""r04: The graph store and entity merge.

Motto: one node per thing, however many times the text mentions it.

Builds a knowledge graph from the validated extractions of r03. The same entity named in many
chunks becomes one node; the same relation stated in many chunks becomes one edge whose weight is
the number of chunks that stated it. Every node and edge remembers which chunks it came from.

Run:  python r04_graph/code.py --offline
      python r04_graph/code.py --top 8
      python r04_graph/code.py --path "Hook" "Memory"
      python r04_graph/code.py --node Subagent
      python r04_graph/code.py --compact
      python r04_graph/code.py --save
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import raka_chunking as ck  # noqa: E402
import raka_common as rc  # noqa: E402
import raka_extract as ex  # noqa: E402
from raka_fake_model import FakeExtractor, fake_summarize  # noqa: E402
from raka_graph import KnowledgeGraph  # noqa: E402

# (label, strategy, size in words, overlap in words)
CONFIGS = [
    ("whole document", "doc", 0, 0),
    ("recursive 40", "recursive", 40, 0),
    ("recursive 20", "recursive", 20, 0),
    ("fixed 40/10", "fixed", 40, 10),
]
DEFAULT_SAVE = ROOT / "index" / "graph.json"


def build_graph(strategy: str = "recursive", size: int = 40, overlap: int = 0, aliases=None) -> KnowledgeGraph:
    chunks = ck.chunk_corpus(rc.load_corpus(), strategy, size, overlap)
    graph = KnowledgeGraph(aliases)
    graph.add_all(ex.extract_corpus(chunks, FakeExtractor()))
    return graph


def print_report() -> None:
    print("\nOne graph per chunking (same fake extraction model as r03).")
    print(f"  {'chunking':<16}{'mentions':>9}{'nodes':>6}{'edges':>6}{'groups':>7}{'isolated':>9}{'avg deg':>8}")
    for label, strategy, size, overlap in CONFIGS:
        s = build_graph(strategy, size, overlap).stats()
        print(f"  {label:<16}{s['mentions']:>9}{s['nodes']:>6}{s['edges']:>6}{s['components']:>7}{s['isolated']:>9}{s['avg_degree']:>8.2f}")
    print("\nmentions = entity mentions across chunks; nodes = distinct entities after the merge; "
          "groups = connected groups of nodes.\nChunking changes how many relations are found (an edge needs two terms in one chunk), "
          "not how many entities.")


def show_top(n: int) -> None:
    graph = build_graph()
    s = graph.stats()
    print(f"Graph from 'recursive 40': {s['nodes']} nodes, {s['edges']} edges. Most connected:")
    ranked = sorted(graph.nodes.values(), key=lambda x: (-graph.degree(x.key), x.name))[:n]
    for node in ranked:
        print(f"  {node.name:<24}degree {graph.degree(node.key):>2}  type {node.type:<10} in {len(node.sources)} chunks")


def show_node(name: str) -> int:
    graph = build_graph()
    key = graph.resolve(name)
    node = graph.nodes.get(key)
    if node is None:
        print(f"no entity named {name!r}. Try: {', '.join(sorted(n.name for n in graph.nodes.values())[:8])} ...")
        return 2
    print(f"{node.name} ({node.type}), in {len(node.sources)} chunks: {', '.join(node.sources[:6])}{' ...' if len(node.sources) > 6 else ''}")
    for d in node.descriptions[:3]:
        print(f"  - {d[:110]}")
    print("  linked to: " + ", ".join(f"{graph.nodes[k].name} (x{graph.edge_between(node.key, k).weight})" for k in graph.neighbours(node.key)))
    return 0


def show_path(a: str, b: str) -> int:
    graph = build_graph()
    path = graph.shortest_path(a, b)
    if path is None:
        print(f"no path between {a!r} and {b!r} (an unknown name, or they are in different groups)")
        return 2
    print(" -> ".join(path) + f"   ({len(path) - 1} hop{'s' if len(path) != 2 else ''})")
    for x, y in zip(path, path[1:]):
        edge = graph.edge_between(x, y)
        print(f"  {x} -- {y} [{', '.join(edge.keywords)}] stated in {edge.weight} chunk(s)")
    return 0


def run_compact() -> int:
    graph = build_graph()
    before = sum(len(n.description.split()) for n in graph.nodes.values())
    calls = graph.compact(fake_summarize, max_fragments=3)
    after = sum(len(n.description.split()) for n in graph.nodes.values())
    print(f"Summary calls made: {calls} (each one is a model call in a real run)")
    print(f"Total node description words: {before} before, {after} after")
    return 0


def main(argv: list[str]) -> int:
    args = list(argv)
    try:
        while args:
            arg = args.pop(0)
            if arg == "--top":
                show_top(int(args.pop(0)) if args and args[0].isdigit() else 5)
                return 0
            if arg == "--node" and args:
                return show_node(args.pop(0))
            if arg == "--path" and len(args) >= 2:
                return show_path(args.pop(0), args.pop(0))
            if arg == "--compact":
                return run_compact()
            if arg == "--save":
                path = Path(args.pop(0)) if args else DEFAULT_SAVE
                build_graph().save(path)
                print(f"saved {path}")
                return 0
            if arg == "--offline":
                continue  # accepted for parity with the Bedrock course; r04 is always offline
            print(f"unknown or incomplete argument: {arg}")
            return 2
    except ValueError as err:
        print(f"error: {err}")
        return 2
    print("rakaRAG r04: graph store and entity merge, offline")
    print_report()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
