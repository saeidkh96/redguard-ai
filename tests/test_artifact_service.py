import json
from pathlib import Path
from redguard.services import ArtifactService


def test_artifact_json_written(tmp_path):
    path = ArtifactService(tmp_path).write_json("abc", "report.json", {"ok": True})
    assert Path(path).exists()
    assert json.loads(Path(path).read_text()) == {"ok": True}
