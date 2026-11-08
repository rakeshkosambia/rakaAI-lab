"""b07: Skill loading.

Motto: load knowledge on demand, not upfront.

Two layers:
  Layer 1 (cheap, always in the system prompt): each skill's name + one-line description.
  Layer 2 (on demand): the full body, returned by the load_skill tool when the model asks.

Why: ten skills of 2,000 words each would cost every single request. With layering,
the model pays only for the skills it actually uses.

Skills live in a source with two methods, names() and read(name):
  LocalSkillSource  the skills/ folder (default)
  S3SkillSource     s3://bucket/prefix/<name>/SKILL.md  (set SKILLS_SOURCE=s3; see iam/s3-read-prefix.json)

Run:  python b07_skill_loading/code.py --offline
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab_common as lc  # noqa: E402
import lab_tools  # noqa: E402

DEFAULT_PROMPT = "Assess the impact of a change request that renames the 'customer_id' field in the orders service. Use the right skill."
SKILLS_DIR = lc.ROOT / "skills"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse a tiny '---' header of 'key: value' lines. Returns (meta, body)."""
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return meta, "\n".join(lines[index + 1:]).strip()
        if ":" in lines[index]:
            key, value = lines[index].split(":", 1)
            meta[key.strip()] = value.strip()
    return {}, text  # header never closed: treat everything as body


class LocalSkillSource:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def names(self) -> list[str]:
        return sorted(p.parent.name for p in self.directory.glob("*/SKILL.md"))

    def read(self, name: str) -> str:
        path = (self.directory / name / "SKILL.md").resolve()
        if self.directory.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(f"no such skill: {name}")
        return path.read_text(encoding="utf-8")


class S3SkillSource:
    """Skills stored as <prefix><name>/SKILL.md in one bucket."""

    def __init__(self, s3_client, bucket: str, prefix: str = "skills/"):
        self.s3 = s3_client
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"

    def names(self) -> list[str]:
        names = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix, Delimiter="/"):
            for entry in page.get("CommonPrefixes", []):
                names.append(entry["Prefix"][len(self.prefix):].strip("/"))
        return sorted(n for n in names if n)

    def read(self, name: str) -> str:
        if "/" in name or ".." in name:
            raise ValueError("invalid skill name")
        body = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}{name}/SKILL.md")["Body"].read()
        return body.decode("utf-8")


class SkillLoader:
    def __init__(self, source):
        self.source = source
        self._catalog: dict[str, str] | None = None

    def catalog(self) -> dict[str, str]:
        """name -> description (layer 1)."""
        if self._catalog is None:
            self._catalog = {}
            for name in self.source.names():
                meta, _ = parse_frontmatter(self.source.read(name))
                self._catalog[meta.get("name", name)] = meta.get("description", "(no description)")
        return self._catalog

    def system_section(self) -> str:
        lines = [f"- {name}: {description}" for name, description in self.catalog().items()]
        return "Skills you can load with the load_skill tool (load one only when it fits the task):\n" + "\n".join(lines)

    def load(self, name: str) -> str:
        """The full skill body (layer 2)."""
        if name not in self.catalog():
            raise ValueError(f"unknown skill '{name}'. Available: {', '.join(sorted(self.catalog())) or '(none)'}")
        _, body = parse_frontmatter(self.source.read(name))
        return f'<skill name="{name}">\n{body}\n</skill>'


def add_skill_tool(registry: lab_tools.ToolRegistry, loader: SkillLoader) -> None:
    registry.register(
        "load_skill",
        "Load the full instructions of a skill by name.",
        {"name": {"type": "string", "description": "Skill name from the skills list"}},
        ["name"],
        lambda args: loader.load(str(args["name"]).strip()),
    )


def build_source(offline: bool):
    if offline or os.getenv("SKILLS_SOURCE", "local").lower() != "s3":
        return LocalSkillSource(SKILLS_DIR)
    bucket = os.getenv("SKILLS_S3_BUCKET", "").strip()
    if not bucket:
        raise SystemExit("SKILLS_SOURCE=s3 needs SKILLS_S3_BUCKET in .env")
    return S3SkillSource(lc.make_session().client("s3"), bucket, os.getenv("SKILLS_S3_PREFIX", "skills/"))


def agent_loop(client, model_id, messages, registry, system, max_turns=8, verbose=True):
    usage: dict = {}
    for turn in range(1, max_turns + 1):
        response = client.converse(
            modelId=model_id,
            messages=messages,
            system=system,
            inferenceConfig={"maxTokens": 1024},
            toolConfig=registry.tool_config(),
        )
        lc.add_usage(usage, response.get("usage"))
        message = response["output"]["message"]
        messages.append(message)
        if response["stopReason"] != "tool_use":
            return lc.message_text(message), response["stopReason"], usage
        results = []
        for call in lc.message_tool_uses(message):
            output, is_error = registry.call(call["name"], call["input"])
            results.append(lc.tool_result_block(call["toolUseId"], output, is_error))
            if verbose:
                print(f"[turn {turn}] {call['name']}({call['input']}){' ERROR' if is_error else ''} -> {len(str(output))} chars")
        messages.append({"role": "user", "content": results})
    return "(stopped: max_turns reached)", "max_turns", usage


def offline_script() -> list:
    return [
        lc.tool_use_response(("load_skill", {"name": "change-request-impact"})),
        lc.text_response(
            "Following the change-request-impact skill: the rename touches every reader and writer of customer_id "
            "(CROSS-CUTTING if it is a stored column). Next I would trace the entry points and list the other callers."
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    lc.load_env()
    args = lc.common_parser(__doc__.splitlines()[0], DEFAULT_PROMPT).parse_args(argv)
    try:
        loader = SkillLoader(build_source(args.offline))
        catalog = loader.catalog()
    except Exception as exc:  # noqa: BLE001
        print("Could not read skills:", lc.explain_aws_error(exc))
        return 1
    print(f"{len(catalog)} skill(s) found: {', '.join(catalog)}")
    system = [{"text": "You are a careful assistant. Keep answers short.\n\n" + loader.system_section()}]
    print(f"System prompt costs about {len(system[0]['text'])} characters for ALL skill descriptions.")
    registry = lab_tools.build_default_registry()
    add_skill_tool(registry, loader)
    client = lc.make_runtime_client(args.offline, offline_script() if args.offline else None)
    model_id = lc.get_model_id(args.offline)
    try:
        text, stop, usage = agent_loop(client, model_id, [lc.user_text(args.prompt)], registry, system, args.max_turns)
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", lc.explain_aws_error(exc))
        return 1
    print("\nAnswer:", text)
    print("Stop:", stop, "| tokens:", usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
