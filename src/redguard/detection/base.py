from abc import ABC, abstractmethod

import numpy as np

from redguard.models.detection import ComponentDetection


class ComponentDetector(ABC):
    """Base interface for component detection backends."""

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[ComponentDetection, ...]:
        """Detect components in an image."""
        raise NotImplementedError
