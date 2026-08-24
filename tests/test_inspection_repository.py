from redguard.persistence import InspectionRecord, InspectionRepository


def test_repository_round_trip(tmp_path):
    repo = InspectionRepository(tmp_path / "history.json")
    record = InspectionRecord("i1", "Q14", "transistor", "FAIL", "CRITICAL", .8, .9, "changed")
    repo.save(record)
    assert repo.get("i1") == record
    assert len(repo.list()) == 1


def test_repository_missing_returns_none(tmp_path):
    assert InspectionRepository(tmp_path / "history.json").get("missing") is None
