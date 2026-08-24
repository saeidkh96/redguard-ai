from __future__ import annotations

import asyncio
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import httpx

from redguard.api.app import create_app
from redguard.core.config import settings
from redguard.persistence import InspectionRepository
from redguard.services import ArtifactService, InspectionService


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.0.0"


async def validate_api() -> None:
    with tempfile.TemporaryDirectory(prefix="redguard-v1-api-") as temp:
        temp_path = Path(temp)
        service = InspectionService(
            InspectionRepository(temp_path / "history.json"),
            ArtifactService(temp_path / "artifacts"),
        )
        app = create_app(service)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://redguard.local",
        ) as client:
            health = await client.get("/health")
            ready = await client.get("/ready")

        assert health.status_code == 200
        assert ready.status_code == 200
        assert health.json()["version"] == EXPECTED_VERSION
        assert ready.json()["status"] == "ready"
        assert "end-to-end-orchestration" in ready.json()["capabilities"]


def validate_metadata() -> None:
    assert settings.version == EXPECTED_VERSION
    installed = importlib.metadata.version("redguard-ai")
    assert installed == EXPECTED_VERSION, (installed, EXPECTED_VERSION)


def validate_files() -> None:
    required = [
        ROOT / "Dockerfile",
        ROOT / "docker-compose.yml",
        ROOT / ".dockerignore",
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / "docs" / "release-v1.0.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert not missing, f"Missing release files: {missing}"


def validate_git_hygiene() -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files = set(result.stdout.splitlines())

    forbidden_exact = {
        "yolo11n.pt",
        "data/detection/labels/train.cache",
        "data/detection/labels/val.cache",
        "data/detection/labels/test.cache",
    }
    bad = sorted(forbidden_exact & files)
    bad.extend(sorted(path for path in files if path.startswith("runs/")))
    bad.extend(sorted(path for path in files if ".egg-info/" in path))
    assert not bad, f"Generated files still tracked: {bad[:20]}"


def validate_wheel_build() -> None:
    with tempfile.TemporaryDirectory(prefix="redguard-v1-wheel-") as temp:
        command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            temp,
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        wheels = list(Path(temp).glob("redguard_ai-1.0.0-*.whl"))
        assert wheels, "v1.0.0 wheel was not generated"


def main() -> int:
    print("RedGuard AI v1.0.0 Release Validation")
    print("=" * 41)

    checks = []

    def run(name, function):
        try:
            function()
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            checks.append(False)
        else:
            print(f"[PASS] {name}")
            checks.append(True)

    run("Release metadata consistent", validate_metadata)
    run("Release files present", validate_files)
    run("Repository hygiene validated", validate_git_hygiene)
    run("Wheel packaging validated", validate_wheel_build)

    try:
        asyncio.run(validate_api())
    except Exception as exc:
        print(f"[FAIL] API health/readiness validated: {exc}")
        checks.append(False)
    else:
        print("[PASS] API health/readiness validated")
        checks.append(True)

    print()
    if all(checks):
        print("REDGUARD v1.0.0 RELEASE VALIDATION: PASS")
        return 0

    print("REDGUARD v1.0.0 RELEASE VALIDATION: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
