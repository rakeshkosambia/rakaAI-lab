"""A fake extraction "model" for offline work. NOT intelligent: a word list plus sentence co-occurrence.

It reads the text between the prompt's markers, finds known terms, and answers in the delimited
format a real model would use, so the same prompt and validator run offline and live.
Faults can be injected to prove the validator catches each kind of bad answer.
"""
from __future__ import annotations

import re

from raka_extract import DONE, TUPLE

# (canonical name, type, pattern)
LEXICON = [
    ("Agent Loop", "concept", r"\bagent loop\b|\bloop\b"),
    ("Tool", "mechanism", r"\btools?\b"),
    ("Permission", "mechanism", r"\bpermissions?\b"),
    ("Hook", "mechanism", r"\bhooks?\b"),
    ("Subagent", "mechanism", r"\bsubagents?\b"),
    ("Context Window", "concept", r"\bcontext window\b|(?<!model )\bcontext\b(?! protocol)"),
    ("Context Compaction", "technique", r"\bcompaction\b"),
    ("Memory", "concept", r"\bmemor(?:y|ies)\b"),
    ("Skill", "mechanism", r"\bskills?\b"),
    ("Model Context Protocol", "protocol", r"\bmodel context protocol\b|\bmcp\b"),
    ("RAG", "technique", r"\brag\b|\bretrieval augmented generation\b"),
    ("Conversation", "concept", r"\bconversation\b"),
    ("IAM Role", "service", r"\biam\b"),
    ("Token Budget", "concept", r"\btoken budget\b"),
    ("BM25", "technique", r"\bbm25\b"),
    ("Embedding", "technique", r"\bembedding\b"),
    ("Graph RAG", "technique", r"\bgraph based rag\b"),
    ("DynamoDB", "service", r"\bdynamodb\b"),
    ("S3", "service", r"\bs3\b"),
    ("Gateway", "service", r"\bgateway\b"),
    ("Audit Log", "artifact", r"\baudit logs?\b|\baudit trail\b"),
]
VERBS = [
    (r"\bprotect", "protects"), (r"\bblock", "blocks"), (r"\blimit", "limits"), (r"\bload", "loads"),
    (r"\bsummaris|\bshrink", "shrinks"), (r"\bsave|\bwrite|\bkeep|\bstore", "stores"), (r"\bstart", "starts"),
    (r"\bcall", "calls"), (r"\bapply|\benforce", "enforces"), (r"\bsearch|\bfetch|\bretriev", "retrieves"),
    (r"\bpublish|\bannounce|\boffer", "offers"),
]
FAULTS = ("short_entity", "placeholder", "bad_prefix", "ghost_relation", "self_relation",
          "duplicate_reversed", "no_complete", "bad_type", "unknown_kind")


def _input_text(prompt: str) -> str:
    match = re.search(r"---Input Text---\n(.*?)\n---End---", prompt, re.S)
    return match.group(1) if match else ""


def _sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.;:])\s+|\n+", text.replace("#", " "))
    return [re.sub(r"\s+", " ", p).strip() for p in pieces if p.strip()]


def _short(sentence: str, limit: int = 160) -> str:
    return sentence[:limit].replace("<|", "").replace("|>", "").strip()


class FakeExtractor:
    """Callable like a model: fake(prompt) -> answer text. Deterministic."""

    def __init__(self, faults: tuple[str, ...] = (), max_entities: int = 6):
        unknown = set(faults) - set(FAULTS)
        if unknown:
            raise ValueError(f"unknown fault(s): {sorted(unknown)}; choose from {FAULTS}")
        self.faults = tuple(faults)
        self.max_entities = max_entities
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        max_rows = int(re.search(r"At most (\d+) rows", prompt).group(1)) if re.search(r"At most (\d+) rows", prompt) else 12
        sentences = _sentences(_input_text(prompt))

        found: dict[str, tuple[str, int, str]] = {}  # name -> (type, order, first sentence)
        per_sentence: list[list[tuple[int, str]]] = []
        for sentence in sentences:
            hits = []
            for name, etype, pattern in LEXICON:
                m = re.search(pattern, sentence, re.I)
                if m:
                    hits.append((m.start(), name))
                    if name not in found:
                        found[name] = (etype, len(found), sentence)
                    elif len(found[name][2].split()) < 4 and len(sentence.split()) >= 4:
                        # a heading such as "Hooks" is a poor description; prefer a real sentence
                        found[name] = (found[name][0], found[name][1], sentence)
            per_sentence.append(sorted(hits))

        keep = {n for n, _ in sorted(found.items(), key=lambda kv: kv[1][1])[: min(self.max_entities, max_rows)]}
        lines = [f"entity{TUPLE}{n}{TUPLE}{found[n][0]}{TUPLE}{_short(found[n][2])}"
                 for n in sorted(keep, key=lambda n: found[n][1])]
        budget = max(0, max_rows - len(lines))

        seen: set[frozenset[str]] = set()
        relations: list[str] = []
        for sentence, hits in zip(sentences, per_sentence):
            names = [n for _, n in hits if n in keep]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pair = frozenset((names[i], names[j]))
                    if names[i] == names[j] or pair in seen or len(relations) >= budget:
                        continue
                    seen.add(pair)
                    words = [kw for pat, kw in VERBS if re.search(pat, sentence, re.I)][:2] or ["related"]
                    relations.append(f"relation{TUPLE}{names[i]}{TUPLE}{names[j]}{TUPLE}{', '.join(words)}{TUPLE}{_short(sentence)}")
        lines += relations
        lines.append(DONE)
        return "\n".join(self._inject(lines))

    # ---------------------------------------------------------------- fault injection
    def _inject(self, lines: list[str]) -> list[str]:
        lines = list(lines)
        ent = [i for i, l in enumerate(lines) if l.startswith("entity")]
        rel = [i for i, l in enumerate(lines) if l.startswith("relation")]
        last = len(lines) - 1  # the completion marker
        for fault in self.faults:
            if fault == "short_entity" and ent:
                lines[ent[0]] = TUPLE.join(lines[ent[0]].split(TUPLE)[:-1])
            elif fault == "bad_type" and ent:
                p = lines[ent[0]].split(TUPLE)
                p[2] = "mech/anism("
                lines[ent[0]] = TUPLE.join(p)
            elif fault == "bad_prefix" and rel:
                lines[rel[0]] = "entity" + lines[rel[0]][len("relation"):]
            elif fault == "placeholder":
                lines.insert(last, f"entity{TUPLE}<entity_name>{TUPLE}<entity_type>{TUPLE}<entity_description>")
            elif fault == "ghost_relation":
                lines.insert(last, f"relation{TUPLE}Quantum Widget{TUPLE}Agent Loop{TUPLE}invented{TUPLE}An invented link.")
            elif fault == "self_relation" and ent:
                name = lines[ent[0]].split(TUPLE)[1]
                lines.insert(last, f"relation{TUPLE}{name}{TUPLE}{name}{TUPLE}self{TUPLE}Itself.")
            elif fault == "duplicate_reversed" and rel:
                p = lines[rel[0]].split(TUPLE)
                lines.insert(last, TUPLE.join([p[0], p[2], p[1]] + p[3:]))
            elif fault == "unknown_kind":
                lines.insert(last, f"node{TUPLE}x{TUPLE}y")
            elif fault == "no_complete":
                lines = [l for l in lines if l != DONE]
            last = len(lines) - 1
        return lines
