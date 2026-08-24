import numpy as np

from redguard.orchestration import (
    ComponentSignals,
    PipelineDiagnostics,
    ProductionInspectionPipeline,
    SignalExtractor,
)
from redguard.persistence import InspectionRepository
from redguard.services import ArtifactService, InspectionService


class FakeExtractor(SignalExtractor):
    def __init__(self, healthy=True):
        self.healthy = healthy

    def extract(self, reference, inspection):
        signals = [
            ComponentSignals(
                component_id="Q14",
                component_type="transistor",
                change_score=0.9,
                fingerprint_similarity=0.3,
                anomaly_score=0.95,
                edge_difference=0.75,
                texture_difference=0.7,
                reference_consensus=0.0,
            )
        ]
        diagnostics = PipelineDiagnostics(
            registration_inlier_ratio=0.85 if self.healthy else 0.05,
            change_area_ratio=0.02,
            changed_region_count=1,
            detection_count=1,
            component_count=1,
        )
        return signals, diagnostics


def build_service(tmp_path):
    return InspectionService(
        InspectionRepository(tmp_path / "history.json"),
        ArtifactService(tmp_path / "artifacts"),
    )


def test_pipeline_persists_component_findings(tmp_path):
    pipeline = ProductionInspectionPipeline(
        service=build_service(tmp_path),
        signal_extractor=FakeExtractor(),
    )
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    result = pipeline.run(image, image)
    assert result.quality_passed
    assert len(result.inspection_ids) == 1
    assert result.decisions["Q14"] == "FAIL"


def test_pipeline_fails_closed_on_bad_visual_quality(tmp_path):
    service = build_service(tmp_path)
    pipeline = ProductionInspectionPipeline(
        service=service,
        signal_extractor=FakeExtractor(healthy=False),
    )
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    result = pipeline.run(image, image)
    assert not result.quality_passed
    assert result.inspection_ids == []
    assert service.history() == []
