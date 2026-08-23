import numpy as np
import pytest

from redguard.core.exceptions import ImageValidationError
from redguard.imaging.preprocessor import (
    ImagePreprocessor,
    PreprocessingConfig,
)


def test_default_preprocessing_returns_grayscale():
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    result = ImagePreprocessor().preprocess(image)

    assert result.shape == (100, 120)
    assert result.dtype == np.uint8


def test_preprocessing_does_not_modify_input():
    image = np.random.default_rng(42).integers(
        0,
        256,
        size=(100, 120, 3),
        dtype=np.uint8,
    )
    original = image.copy()

    ImagePreprocessor().preprocess(image)

    assert np.array_equal(image, original)


def test_resize_preserves_aspect_ratio_by_width():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    config = PreprocessingConfig(
        target_width=100,
        denoise=False,
        normalize_contrast=False,
    )

    result = ImagePreprocessor(config).preprocess(image)

    assert result.shape == (50, 100)


def test_resize_preserves_aspect_ratio_by_height():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    config = PreprocessingConfig(
        target_height=50,
        denoise=False,
        normalize_contrast=False,
    )

    result = ImagePreprocessor(config).preprocess(image)

    assert result.shape == (50, 100)


def test_resize_fits_inside_width_and_height():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    config = PreprocessingConfig(
        target_width=100,
        target_height=100,
        denoise=False,
        normalize_contrast=False,
    )

    result = ImagePreprocessor(config).preprocess(image)

    assert result.shape == (50, 100)


def test_color_can_be_preserved():
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    config = PreprocessingConfig(
        grayscale=False,
        denoise=False,
        normalize_contrast=False,
    )

    result = ImagePreprocessor(config).preprocess(image)

    assert result.shape == (100, 120, 3)


def test_preprocessing_is_deterministic():
    image = np.random.default_rng(7).integers(
        0,
        256,
        size=(100, 120, 3),
        dtype=np.uint8,
    )

    preprocessor = ImagePreprocessor()

    first = preprocessor.preprocess(image)
    second = preprocessor.preprocess(image)

    assert np.array_equal(first, second)


def test_invalid_image_is_rejected():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ImageValidationError):
        ImagePreprocessor().preprocess(image)
