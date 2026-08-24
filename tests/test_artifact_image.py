import cv2
import numpy as np

from redguard.services import ArtifactService


def test_artifact_service_writes_image(tmp_path):
    service = ArtifactService(tmp_path / "artifacts")
    image = np.full((20, 30, 3), 127, dtype=np.uint8)
    path = service.write_image("inspection-1", "crop.png", image)
    loaded = cv2.imread(path)
    assert loaded is not None
    assert loaded.shape == image.shape
