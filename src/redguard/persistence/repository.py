from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from .models import InspectionRecord


class InspectionRepository:
    """Small JSON persistence layer with atomic writes for local production validation."""

    def __init__(self, path: str | Path = "artifacts/inspection_history.json") -> None:
        self.path = Path(path)
        self._lock = RLock()

    def save(self, record: InspectionRecord) -> InspectionRecord:
        with self._lock:
            records = {item.inspection_id: item for item in self.list()}
            records[record.inspection_id] = record
            self._write(list(records.values()))
        return record

    def get(self, inspection_id: str) -> InspectionRecord | None:
        return next((x for x in self.list() if x.inspection_id == inspection_id), None)

    def list(self) -> list[InspectionRecord]:
        with self._lock:
            if not self.path.exists():
                return []
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [InspectionRecord.from_dict(item) for item in data]

    def _write(self, records: list[InspectionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps([r.to_dict() for r in records], indent=2), encoding="utf-8")
        temp.replace(self.path)
