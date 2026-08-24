from redguard.orchestration import PipelineDiagnostics, PipelineQualityGate


def diagnostics(**overrides):
    values = dict(
        registration_inlier_ratio=0.85,
        change_area_ratio=0.02,
        changed_region_count=1,
        detection_count=4,
        component_count=4,
    )
    values.update(overrides)
    return PipelineDiagnostics(**values)


def test_quality_gate_accepts_healthy_pipeline():
    result = PipelineQualityGate().evaluate(diagnostics())
    assert result.passed
    assert result.messages == ()


def test_quality_gate_rejects_bad_registration():
    result = PipelineQualityGate().evaluate(
        diagnostics(registration_inlier_ratio=0.05)
    )
    assert not result.passed
    assert "registration quality below threshold" in result.messages


def test_quality_gate_rejects_missing_detections():
    result = PipelineQualityGate().evaluate(
        diagnostics(detection_count=0, component_count=0)
    )
    assert not result.passed
