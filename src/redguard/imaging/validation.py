import numpy as np

from redguard.core.config import settings
from redguard.core.exceptions import ImageValidationError


def validate_image(image: np.ndarray) -> None:
    """Validate an image before it enters the RedGuard pipeline."""

    if not isinstance(image, np.ndarray):
        raise ImageValidationError("Image must be a NumPy array.")

    if image.size == 0:
        raise ImageValidationError("Image is empty.")

    if image.ndim not in (2, 3):
        raise ImageValidationError(
            f"Unsupported image dimensions: {image.ndim}"
        )

    height, width = image.shape[:2]

    if width < settings.min_image_width:
        raise ImageValidationError(
            f"Image width {width} is below minimum "
            f"{settings.min_image_width}."
        )

    if height < settings.min_image_height:
        raise ImageValidationError(
            f"Image height {height} is below minimum "
            f"{settings.min_image_height}."
        )

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ImageValidationError(
            f"Unsupported channel count: {image.shape[2]}"
        )
