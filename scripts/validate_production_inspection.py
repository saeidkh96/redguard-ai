from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from redguard.api import create_app
from redguard.persistence import InspectionRepository
from redguard.services import ArtifactService, InspectionService


def main() -> None:
    print("RedGuard AI Production Inspection Validation")
    print("=" * 46)
    with TemporaryDirectory() as temp:
        root = Path(temp)
        service = InspectionService(InspectionRepository(root / "history.json"), ArtifactService(root / "artifacts"))
        client = TestClient(create_app(service))
        payload = {"component_id":"Q14","component_type":"transistor","change_score":.9,
            "fingerprint_similarity":.3,"anomaly_score":.95,"edge_difference":.75,
            "texture_difference":.7,"reference_consensus":0.0}
        response = client.post("/api/v1/inspections", json=payload)
        body = response.json()
        checks = [
            (response.status_code == 201, "Inspection API accepted request"),
            (body["decision"] == "FAIL", "Deterministic inspection decision preserved"),
            (bool(body["explanation"]), "Safe reasoning explanation attached"),
            (len(service.history()) == 1, "Inspection history persisted"),
            (Path(body["artifacts"][0]).exists(), "Structured report artifact persisted"),
            (client.get(f'/api/v1/inspections/{body["inspection_id"]}').status_code == 200, "Inspection retrievable by ID"),
        ]
        for ok, message in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {message}")
        if not all(ok for ok, _ in checks):
            raise SystemExit("REDGUARD v0.9.0 PRODUCTION INSPECTION: FAIL")
    print("\nREDGUARD v0.9.0 PRODUCTION INSPECTION: PASS")


if __name__ == "__main__":
    main()
