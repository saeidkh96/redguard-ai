class RedGuardError(Exception):
    """Base exception for RedGuard AI."""


class ImageLoadError(RedGuardError):
    """Raised when an image cannot be loaded."""


class ImageValidationError(RedGuardError):
    """Raised when an image fails validation."""
