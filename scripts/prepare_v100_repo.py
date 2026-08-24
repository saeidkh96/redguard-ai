from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        print(f"Removed directory: {path.relative_to(ROOT)}")
    elif path.exists():
        path.unlink()
        print(f"Removed file: {path.relative_to(ROOT)}")


def main() -> None:
    # Generated training output is not part of the curated v1 working tree.
    remove(ROOT / "runs")
    remove(ROOT / "yolo11n.pt")

    for cache in (ROOT / "data" / "detection" / "labels").glob("*.cache"):
        remove(cache)

    for egg_info in (ROOT / "src").glob("*.egg-info"):
        remove(egg_info)

    for cache_dir in ROOT.rglob("__pycache__"):
        if ".venv" not in cache_dir.parts:
            remove(cache_dir)

    print("Repository runtime/generated artifacts cleaned for v1.0.0")


if __name__ == "__main__":
    main()
