from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from redguard.anomaly.patch_memory import (
    PatchMemoryAnomalyDetector,
)
from redguard.features.fingerprint import (
    ComponentFingerprinter,
)


class InspectionDecision(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


@dataclass(frozen=True)
class FineGrainedInspectionResult:
    fingerprint_similarity: float
    anomaly_score: float
    edge_difference: float
    texture_difference: float
    risk_score: float
    decision: InspectionDecision


class FineGrainedInspector:
    """
    Fuse identity, anomaly, edge and texture signals
    into a fine-grained component inspection result.
    """

    def __init__(
        self,
        fingerprinter: ComponentFingerprinter,
        anomaly_detector: PatchMemoryAnomalyDetector | None = None,
        pass_threshold: float = 0.25,
        fail_threshold: float = 0.55,
        anomaly_scale: float = 0.20,
    ) -> None:
        if not (
            0 <= pass_threshold
            < fail_threshold
            <= 1
        ):
            raise ValueError(
                "Expected 0 <= pass_threshold "
                "< fail_threshold <= 1."
            )

        self.fingerprinter = fingerprinter
        self.anomaly_detector = anomaly_detector
        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold
        self.anomaly_scale = anomaly_scale

    def inspect(
        self,
        reference: np.ndarray,
        inspection: np.ndarray,
    ) -> FineGrainedInspectionResult:
        if reference.shape[:2] != inspection.shape[:2]:
            inspection = cv2.resize(
                inspection,
                (
                    reference.shape[1],
                    reference.shape[0],
                ),
            )

        fingerprint = (
            self.fingerprinter.compare(
                reference,
                inspection,
            )
        )

        edge_difference = (
            self._edge_difference(
                reference,
                inspection,
            )
        )

        texture_difference = (
            self._texture_difference(
                reference,
                inspection,
            )
        )

        anomaly_score = 0.0

        if self.anomaly_detector is not None:
            anomaly_score = (
                self.anomaly_detector
                .predict(inspection)
                .score
            )

        fingerprint_risk = float(
            np.clip(
                (1.0 - fingerprint.similarity)
                / 0.20,
                0.0,
                1.0,
            )
        )

        edge_risk = float(
            np.clip(
                edge_difference / 0.15,
                0.0,
                1.0,
            )
        )

        texture_risk = float(
            np.clip(
                texture_difference / 0.15,
                0.0,
                1.0,
            )
        )

        anomaly_risk = float(
            np.clip(
                anomaly_score
                / self.anomaly_scale,
                0.0,
                1.0,
            )
        )

        risk_score = (
            0.35 * fingerprint_risk
            + 0.25 * anomaly_risk
            + 0.20 * edge_risk
            + 0.20 * texture_risk
        )

        if risk_score < self.pass_threshold:
            decision = InspectionDecision.PASS
        elif risk_score < self.fail_threshold:
            decision = InspectionDecision.REVIEW
        else:
            decision = InspectionDecision.FAIL

        return FineGrainedInspectionResult(
            fingerprint_similarity=(
                fingerprint.similarity
            ),
            anomaly_score=anomaly_score,
            edge_difference=edge_difference,
            texture_difference=texture_difference,
            risk_score=float(risk_score),
            decision=decision,
        )

    @staticmethod
    def _gray(
        image: np.ndarray,
    ) -> np.ndarray:
        if image.ndim == 2:
            return image

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    def _edge_difference(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        first_edges = cv2.Canny(
            self._gray(first),
            60,
            140,
        )

        second_edges = cv2.Canny(
            self._gray(second),
            60,
            140,
        )

        difference = cv2.absdiff(
            first_edges,
            second_edges,
        )

        return float(
            np.count_nonzero(difference)
            / difference.size
        )

    def _texture_difference(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        first_gray = self._gray(
            first
        ).astype(np.float32)

        second_gray = self._gray(
            second
        ).astype(np.float32)

        first_lap = cv2.Laplacian(
            first_gray,
            cv2.CV_32F,
        )

        second_lap = cv2.Laplacian(
            second_gray,
            cv2.CV_32F,
        )

        difference = np.mean(
            np.abs(
                first_lap
                - second_lap
            )
        )

        return float(
            np.clip(
                difference / 255.0,
                0.0,
                1.0,
            )
        )
