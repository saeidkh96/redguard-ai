from redguard.persistence import InspectionRepository
from redguard.services import ArtifactService, InspectionService


def service(tmp_path):
    return InspectionService(InspectionRepository(tmp_path / "history.json"), ArtifactService(tmp_path / "artifacts"))


def test_service_persists_fail(tmp_path):
    result = service(tmp_path).inspect(component_id="Q14", component_type="transistor", change_score=.9,
        fingerprint_similarity=.3, anomaly_score=.95, edge_difference=.75, texture_difference=.7,
        reference_consensus=0.0)
    assert result.decision == "FAIL"
    assert result.explanation
    assert result.artifacts


def test_service_persists_pass(tmp_path):
    result = service(tmp_path).inspect(component_id="R27", component_type="resistor", change_score=.01,
        fingerprint_similarity=.995, anomaly_score=.04, edge_difference=.01, texture_difference=.02,
        reference_consensus=1.0)
    assert result.decision == "PASS"
    assert len(service(tmp_path).history()) == 1
