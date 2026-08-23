import sys

import cv2
import numpy as np

from redguard.core.config import settings
from redguard.imaging.validation import validate_image


def check(label: str, fn) -> bool:
    try:
        fn()
        print(f"[PASS] {label}")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        return False


def main() -> int:
    print("RedGuard AI Foundation Validation")
    print("=" * 35)

    checks = [
        ("OpenCV import", lambda: cv2.__version__),
        ("NumPy import", lambda: np.__version__),
        ("Configuration", lambda: settings.app_name),
        (
            "Image validation",
            lambda: validate_image(
                np.zeros((100, 100, 3), dtype=np.uint8)
            ),
        ),
    ]

    results = [check(label, fn) for label, fn in checks]

    print()

    if all(results):
        print(f"REDGUARD v{settings.version}: PASS")
        return 0

    print(f"REDGUARD v{settings.version}: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
