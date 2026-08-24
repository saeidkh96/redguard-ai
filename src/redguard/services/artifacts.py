from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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

    def write_image(self, inspection_id: str, name: str, image: np.ndarray) -> str:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty NumPy array")
        path = self.inspection_dir(inspection_id) / name
        if not cv2.imwrite(str(path), image):
            raise OSError(f"failed to write image artifact: {path}")
        return str(path)
