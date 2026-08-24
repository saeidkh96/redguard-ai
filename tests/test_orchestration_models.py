import pytest

from redguard.orchestration import ComponentSignals


def test_component_signals_reject_out_of_range_values():
    with pytest.raises(ValueError):
        ComponentSignals(
            component_id="Q14",
            component_type="transistor",
            change_score=1.2,
            fingerprint_similarity=0.9,
            anomaly_score=0.1,
            edge_difference=0.1,
            texture_difference=0.1,
            reference_consensus=1.0,
        )
