from pathlib import Path

from redguard.api.app import create_app
from redguard.core.config import settings


def test_v1_release_version_is_consistent():
    assert settings.version == "1.0.0"
    assert create_app().version == "1.0.0"


def test_release_container_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "Dockerfile").is_file()
    assert (root / "docker-compose.yml").is_file()
    assert (root / ".github" / "workflows" / "ci.yml").is_file()
