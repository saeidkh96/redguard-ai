from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactService:
    def __init__(self, root: str | Path = "artifacts/inspections") -> None:
        self.root = Path(root)

    def inspection_dir(self, inspection_id: str) -> Path:
        path = self.root / inspection_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, inspection_id: str, name: str, payload: dict[str, Any]) -> str:
        path = self.inspection_dir(inspection_id) / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)
