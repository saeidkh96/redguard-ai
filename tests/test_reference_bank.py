import numpy as np
import pytest

from redguard.reference import ReferenceBank, ReferenceSample


def sample(
    sample_id: str,
    component_id: str = "Q14",
    component_type: str = "transistor",
) -> ReferenceSample:
    return ReferenceSample(
        sample_id=sample_id,
        component_id=component_id,
        component_type=component_type,
        fingerprint=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )


def test_reference_bank_groups_samples_by_component():
    bank = ReferenceBank()

    bank.add(sample("ref_001"))
    bank.add(sample("ref_002"))

    reference_set = bank.get("Q14")

    assert reference_set.size == 2
    assert bank.component_count == 1
    assert bank.sample_count == 2


def test_reference_bank_supports_multiple_components():
    bank = ReferenceBank()

    bank.add(sample("q14_ref", "Q14", "transistor"))
    bank.add(sample("r27_ref", "R27", "resistor"))

    assert bank.component_count == 2
    assert bank.sample_count == 2
    assert bank.contains("Q14")
    assert bank.contains("R27")


def test_reference_bank_rejects_duplicate_sample_ids():
    bank = ReferenceBank()

    bank.add(sample("ref_001"))

    with pytest.raises(ValueError):
        bank.add(sample("ref_001"))


def test_reference_bank_rejects_component_type_mismatch():
    bank = ReferenceBank()

    bank.add(sample("ref_001", "Q14", "transistor"))

    with pytest.raises(ValueError):
        bank.add(sample("ref_002", "Q14", "resistor"))


def test_reference_bank_missing_component_raises():
    bank = ReferenceBank()

    with pytest.raises(KeyError):
        bank.get("UNKNOWN")