from fastapi.testclient import TestClient
from redguard.api import create_app
from redguard.persistence import InspectionRepository
from redguard.services import ArtifactService, InspectionService


def client(tmp_path):
    svc = InspectionService(InspectionRepository(tmp_path / "history.json"), ArtifactService(tmp_path / "artifacts"))
    return TestClient(create_app(svc))


def payload():
    return {"component_id":"Q14","component_type":"transistor","change_score":.9,
        "fingerprint_similarity":.3,"anomaly_score":.95,"edge_difference":.75,
        "texture_difference":.7,"reference_consensus":0.0}


def test_health(tmp_path):
    assert client(tmp_path).get("/health").status_code == 200


def test_create_get_and_list(tmp_path):
    c = client(tmp_path)
    created = c.post("/api/v1/inspections", json=payload())
    assert created.status_code == 201
    body = created.json()
    assert body["decision"] == "FAIL"
    assert c.get(f'/api/v1/inspections/{body["inspection_id"]}').status_code == 200
    assert len(c.get("/api/v1/inspections").json()) == 1


def test_missing_is_404(tmp_path):
    assert client(tmp_path).get("/api/v1/inspections/missing").status_code == 404
