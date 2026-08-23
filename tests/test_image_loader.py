import cv2
import numpy as np
import pytest

from redguard.core.exceptions import ImageLoadError
from redguard.imaging.loader import ImageLoader


def test_load_valid_image(tmp_path):
    path = tmp_path / "sample.png"

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)

    loaded = ImageLoader().load(path)

    assert loaded.shape == (100, 120, 3)
    assert loaded.dtype == np.uint8


def test_missing_image():
    with pytest.raises(ImageLoadError):
        ImageLoader().load("does-not-exist.png")


def test_unsupported_extension(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("not an image")

    with pytest.raises(ImageLoadError):
        ImageLoader().load(path)


def test_corrupted_image(tmp_path):
    path = tmp_path / "corrupted.png"
    path.write_bytes(b"this-is-not-a-real-image")

    with pytest.raises(ImageLoadError):
        ImageLoader().load(path)
