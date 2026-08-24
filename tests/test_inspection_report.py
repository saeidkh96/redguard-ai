from redguard.intelligence import (
    InspectionDecision,
    InspectionIntelligenceEngine,
    InspectionReportBuilder,
)


def build_findings():
    engine = InspectionIntelligenceEngine()

    normal = engine.inspect(
        component_id="R27",
        component_type="resistor",
        change_score=0.01,
        fingerprint_similarity=0.995,
        anomaly_score=0.04,
        edge_difference=0.01,
        texture_difference=0.02,
        reference_consensus=1.0,
    )

    altered = engine.inspect(
        component_id="Q14",
        component_type="transistor",
        change_score=0.90,
        fingerprint_similarity=0.30,
        anomaly_score=0.95,
        edge_difference=0.75,
        texture_difference=0.70,
        reference_consensus=0.0,
    )

    return [normal, altered]


def test_report_summarizes_component_decisions():
    report = InspectionReportBuilder().build(build_findings())

    assert report.component_count == 2
    assert report.passed_count == 1
    assert report.failed_count == 1
    assert report.review_count == 0


def test_report_serializes_to_dictionary():
    builder = InspectionReportBuilder()
    report = builder.build(build_findings())

    payload = builder.to_dict(report)

    assert payload["summary"]["components"] == 2
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["failed"] == 1
    assert len(payload["findings"]) == 2

    decisions = {
        item["decision"]
        for item in payload["findings"]
    }

    assert InspectionDecision.PASS.value in decisions
    assert InspectionDecision.FAIL.value in decisions