"""Hygiene checks: no secrets, console-safe text, templates are placeholders, no client data."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".bat", ".example", ".gitignore"}
SKIP_PARTS = {"__pycache__", ".pytest_cache", "logs", "out", "index"}


def repo_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in SKIP_PARTS for part in path.parts):
            if path.suffix in TEXT_SUFFIXES or path.name in {".env.example", ".gitignore"}:
                yield path


def test_no_env_file_is_present():
    assert not (ROOT / ".env").exists()


def test_no_secrets_in_any_file():
    patterns = {
        "aws access key id": re.compile(r"\b(AK" + r"IA|AS" + r"IA)[0-9A-Z]{16}\b"),
        "anthropic style key": re.compile(r"sk-" + r"ant-[A-Za-z0-9_\-]{20,}"),
        "generic api key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
        "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "bearer token": re.compile(r"Bearer [A-Za-z0-9._\-]{30,}"),
    }
    offenders = []
    for path in repo_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)}: {label}")
    assert not offenders, offenders


def test_no_email_addresses_in_the_corpus_or_questions():
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    offenders = [str(p.relative_to(ROOT)) for p in list((ROOT / "corpus").glob("*.md")) + [ROOT / "questions.json"]
                 if email.search(p.read_text(encoding="utf-8"))]
    assert not offenders, offenders


def test_all_text_files_are_ascii_so_windows_consoles_are_safe():
    non_ascii = []
    for path in repo_files():
        try:
            path.read_text(encoding="ascii")
        except UnicodeDecodeError:
            non_ascii.append(str(path.relative_to(ROOT)))
    assert not non_ascii, non_ascii


def test_env_example_has_no_filled_in_values_except_the_token_cap():
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    for key in ("RAKA_PROVIDER", "RAKA_EXTRACT_MODEL", "RAKA_QUERY_MODEL", "RAKA_EMBED_MODEL"):
        assert values[key] == "", key


def test_gitignore_protects_env_and_output():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for needed in (".env", "logs/", "__pycache__/"):
        assert needed in text, needed


def test_every_chapter_folder_has_a_readme_and_code():
    for folder in sorted(ROOT.glob("r[0-9][0-9]_*")):
        assert (folder / "README.md").is_file(), folder.name
        assert (folder / "code.py").is_file(), folder.name


def test_no_client_data_paths_are_referenced():
    banned = ("clients\\", "clients/", "client-0", "rakesh-org-secrets")
    offenders = []
    for path in repo_files():
        if path.name == "test_repo_hygiene.py":
            continue
        text = path.read_text(encoding="utf-8")
        for word in banned:
            if word in text:
                offenders.append(f"{path.relative_to(ROOT)}: {word}")
    assert not offenders, offenders
