"""Hygiene checks: no secrets, no account identifiers, console-safe text, templates are placeholders."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".bat", ".example", ".gitignore"}


def repo_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in {"__pycache__", ".pytest_cache", "logs", "out"} for part in path.parts):
            if path.suffix in TEXT_SUFFIXES or path.name in {".env.example", ".gitignore"}:
                yield path


def test_no_env_file_is_present():
    assert not (ROOT / ".env").exists()


def test_no_secrets_or_account_ids_in_any_file():
    patterns = {
        "aws access key id": re.compile(r"\b(AK" + r"IA|AS" + r"IA)[0-9A-Z]{16}\b"),
        "anthropic style key": re.compile(r"sk-" + r"ant-[A-Za-z0-9_\-]{20,}"),
        "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "12 digit account id in an arn": re.compile(r"arn:aws:[a-z0-9\-]*:[a-z0-9\-]*:\d{12}:"),
        "sso start url": re.compile(r"awsapps\.com/start"),
    }
    offenders = []
    for path in repo_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)}: {label}")
    assert not offenders, offenders


def test_all_text_files_are_ascii_so_windows_consoles_are_safe():
    non_ascii = []
    for path in repo_files():
        try:
            path.read_text(encoding="ascii")
        except UnicodeDecodeError:
            non_ascii.append(str(path.relative_to(ROOT)))
    assert not non_ascii, non_ascii


def test_env_example_has_no_filled_in_credentials():
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    for key in ("AWS_PROFILE", "BEDROCK_MODEL_ID", "BEDROCK_GUARDRAIL_ID", "SKILLS_S3_BUCKET", "LAB_PRICE_IN_PER_1K", "LAB_PRICE_OUT_PER_1K"):
        assert values[key] == "", key


def test_gitignore_protects_env_and_output():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for needed in (".env", "logs/", "workspace/out/", "__pycache__/"):
        assert needed in text


def test_iam_templates_are_valid_json_with_placeholders_only():
    for name in ("bedrock-invoke-minimal.json", "s3-read-prefix.json"):
        data = json.loads((ROOT / "iam" / name).read_text(encoding="utf-8"))
        assert data["Version"] == "2012-10-17"
        text = json.dumps(data)
        assert "*:*" not in text and '"Action": "*"' not in text
    invoke = json.loads((ROOT / "iam" / "bedrock-invoke-minimal.json").read_text(encoding="utf-8"))
    assert "ACCOUNT_ID" in json.dumps(invoke) and "MODEL_ID" in json.dumps(invoke)


def test_every_chapter_has_readme_and_code_and_a_motto():
    for folder in sorted(ROOT.glob("b[0-9][0-9]_*")):
        code = folder / "code.py"
        readme = folder / "README.md"
        assert code.is_file() and readme.is_file(), folder.name
        first_lines = readme.read_text(encoding="utf-8").splitlines()[:4]
        assert any(line.startswith("> *") for line in first_lines), f"{folder.name} README needs a motto line"
        assert "--offline" in code.read_text(encoding="utf-8"), folder.name
