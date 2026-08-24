from fastapi.testclient import TestClient

from redguard.api.app import create_app
from redguard.persistence import InspectionRepository
from redguard.services import ArtifactService, InspectionService


def test_readiness_endpoint_reports_orchestration(tmp_path):
    service = InspectionService(
        InspectionRepository(tmp_path / "history.json"),
        ArtifactService(tmp_path / "artifacts"),
    )
    client = TestClient(create_app(service))
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert "end-to-end-orchestration" in body["capabilities"]
