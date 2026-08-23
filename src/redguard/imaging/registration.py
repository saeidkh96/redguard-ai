from dataclasses import dataclass

import cv2
import numpy as np

from redguard.core.exceptions import ImageValidationError
from redguard.imaging.validation import validate_image


@dataclass(frozen=True)
class RegistrationResult:
    aligned_image: np.ndarray
    homography: np.ndarray
    matches_total: int
    matches_used: int
    inliers: int
    inlier_ratio: float


class ImageRegistrationEngine:
    """Align inspection images to a reference using ORB + KNN + RANSAC."""

    def __init__(
        self,
        max_features: int = 5000,
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 3.0,
        min_matches: int = 8,
    ) -> None:
        self.max_features = max_features
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches

    def register(
        self,
        reference: np.ndarray,
        inspection: np.ndarray,
    ) -> RegistrationResult:
        validate_image(reference)
        validate_image(inspection)

        reference_gray = self._to_grayscale(reference)
        inspection_gray = self._to_grayscale(inspection)

        orb = cv2.ORB_create(
            nfeatures=self.max_features,
            scaleFactor=1.2,
            nlevels=8,
        )

        keypoints_ref, descriptors_ref = orb.detectAndCompute(
            reference_gray,
            None,
        )
        keypoints_ins, descriptors_ins = orb.detectAndCompute(
            inspection_gray,
            None,
        )

        if descriptors_ref is None or descriptors_ins is None:
            raise ImageValidationError(
                "Insufficient visual features for registration."
            )

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

        raw_matches = matcher.knnMatch(
            descriptors_ins,
            descriptors_ref,
            k=2,
        )

        good_matches = []

        for match_pair in raw_matches:
            if len(match_pair) != 2:
                continue

            first, second = match_pair

            if first.distance < self.ratio_threshold * second.distance:
                good_matches.append(first)

        if len(good_matches) < self.min_matches:
            raise ImageValidationError(
                f"Not enough reliable feature matches: "
                f"{len(good_matches)} found, "
                f"{self.min_matches} required."
            )

        points_inspection = np.float32(
            [
                keypoints_ins[match.queryIdx].pt
                for match in good_matches
            ]
        ).reshape(-1, 1, 2)

        points_reference = np.float32(
            [
                keypoints_ref[match.trainIdx].pt
                for match in good_matches
            ]
        ).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(
            points_inspection,
            points_reference,
            cv2.RANSAC,
            self.ransac_threshold,
        )

        if homography is None or mask is None:
            raise ImageValidationError(
                "Homography estimation failed."
            )

        inliers = int(mask.ravel().sum())

        if inliers < 4:
            raise ImageValidationError(
                "Registration produced too few geometric inliers."
            )

        inlier_ratio = inliers / len(good_matches)

        height, width = reference.shape[:2]

        aligned = cv2.warpPerspective(
            inspection,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
        )

        return RegistrationResult(
            aligned_image=aligned,
            homography=homography,
            matches_total=len(raw_matches),
            matches_used=len(good_matches),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
        )

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image

        if image.shape[2] == 3:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        if image.shape[2] == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY,
            )

        raise ImageValidationError(
            "Unsupported image channel configuration."
        )
