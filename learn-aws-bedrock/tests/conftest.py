import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_chapter(folder: str):
    """Import a chapter's code.py by folder name (folder names are not valid module names)."""
    path = ROOT / folder / "code.py"
    spec = importlib.util.spec_from_file_location(f"chapter_{folder}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up here
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def chapter():
    return load_chapter


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Keep tests away from real AWS settings and from the repo's workspace folder."""
    for name in ("AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "BEDROCK_MODEL_ID", "BEDROCK_GUARDRAIL_ID", "SKILLS_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAB_WORKDIR", str(tmp_path / "workspace"))
    yield
