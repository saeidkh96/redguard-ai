from pathlib import Path


def test_gitignore_excludes_generated_release_artifacts():
    root = Path(__file__).resolve().parents[1]
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert "runs/" in text
    assert "*.cache" in text
    assert "*.egg-info/" in text
    assert "*.pt" in text
    assert "!models/*.pt" in text


def test_dockerignore_excludes_local_state():
    root = Path(__file__).resolve().parents[1]
    text = (root / ".dockerignore").read_text(encoding="utf-8")
    assert ".venv" in text
    assert "artifacts" in text
    assert "runs" in text


def test_release_document_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "release-v1.0.md").is_file()
