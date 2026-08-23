import numpy as np
import pytest

from redguard.core.exceptions import ImageValidationError
from redguard.imaging.validation import validate_image


def test_valid_rgb_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    validate_image(image)


def test_rejects_non_numpy_input():
    with pytest.raises(ImageValidationError):
        validate_image("not-an-image")


def test_rejects_empty_image():
    image = np.array([], dtype=np.uint8)

    with pytest.raises(ImageValidationError):
        validate_image(image)


def test_rejects_small_image():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ImageValidationError):
        validate_image(image)


def test_rejects_invalid_channel_count():
    image = np.zeros((100, 100, 5), dtype=np.uint8)

    with pytest.raises(ImageValidationError):
        validate_image(image)
