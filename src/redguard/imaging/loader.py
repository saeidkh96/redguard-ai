from pathlib import Path

import cv2
import numpy as np

from redguard.core.exceptions import ImageLoadError
from redguard.imaging.validation import validate_image


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


class ImageLoader:
    """Load and validate images entering the RedGuard pipeline."""

    def load(self, path: str | Path) -> np.ndarray:
        image_path = Path(path)

        if not image_path.exists():
            raise ImageLoadError(
                f"Image does not exist: {image_path}"
            )

        if not image_path.is_file():
            raise ImageLoadError(
                f"Image path is not a file: {image_path}"
            )

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ImageLoadError(
                f"Unsupported image format: {image_path.suffix}"
            )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ImageLoadError(
                f"OpenCV could not decode image: {image_path}"
            )

        validate_image(image)

        return image
