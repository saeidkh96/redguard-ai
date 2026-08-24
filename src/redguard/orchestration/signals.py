from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from redguard.anomaly.patch_memory import PatchMemoryAnomalyDetector
from redguard.detection.baseline import BaselineComponentDetector
from redguard.detection.base import ComponentDetector
from redguard.detection.registry import build_component_registry
from redguard.features.backbone import VisionBackbone
from redguard.features.fingerprint import ComponentFingerprinter
from redguard.imaging.change_detection import ChangeDetector
from redguard.imaging.registration import ImageRegistrationEngine
from redguard.inspection.component_verifier import ComponentVerifier
from redguard.inspection.fine_grained import FineGrainedInspector
from redguard.inspection.multi_reference import MultiReferenceVerifier
from redguard.orchestration.models import ComponentSignals, PipelineDiagnostics
from redguard.reference import ReferenceSample, ReferenceSet


class SignalExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        reference: np.ndarray,
        inspection: np.ndarray,
    ) -> tuple[list[ComponentSignals], PipelineDiagnostics]:
        raise NotImplementedError


class ImageSignalExtractor(SignalExtractor):
    """Connect RedGuard's visual subsystems into component-level signals."""

    def __init__(
        self,
        *,
        detector: ComponentDetector | None = None,
        pretrained_backbone: bool = False,
    ) -> None:
        self.detector = detector or BaselineComponentDetector(
            min_area=500,
            max_area=20000,
        )
        self.registration = ImageRegistrationEngine()
        self.change_detector = ChangeDetector(
            threshold=25,
            ssim_threshold=20,
            min_region_area=80,
            morphology_kernel_size=3,
            comparison_blur_size=5,
            border_margin=12,
        )
        self.component_verifier = ComponentVerifier(min_overlap_ratio=0.05)
        self.backbone = VisionBackbone(pretrained=pretrained_backbone)
        self.fingerprinter = ComponentFingerprinter(
            backbone=self.backbone,
            similarity_threshold=0.97,
        )

    def extract(
        self,
        reference: np.ndarray,
        inspection: np.ndarray,
    ) -> tuple[list[ComponentSignals], PipelineDiagnostics]:
        registration = self.registration.register(reference, inspection)

        change = self.change_detector.detect(
            reference,
            registration.aligned_image,
            valid_mask=registration.valid_mask,
        )

        detections = self.detector.detect(reference)
        registry = build_component_registry(detections)

        verification = self.component_verifier.verify(
            registry.components,
            change.regions,
        )
        verification_by_id = {
            item.component_id: item for item in verification.components
        }

        signals: list[ComponentSignals] = []

        for component in registry.components:
            reference_crop = self._crop(reference, component.bounding_box)
            inspection_crop = self._crop(
                registration.aligned_image,
                component.bounding_box,
            )

            variants = self._normal_variants(reference_crop)

            reference_set = ReferenceSet(
                component_id=component.component_id,
                component_type=component.component_type,
            )

            for index, variant in enumerate(variants, start=1):
                reference_set.add(
                    ReferenceSample(
                        sample_id=f"normal_{index:03d}",
                        component_id=component.component_id,
                        component_type=component.component_type,
                        fingerprint=self.fingerprinter.fingerprint(variant),
                    )
                )

            inspection_fingerprint = self.fingerprinter.fingerprint(
                inspection_crop
            )

            reference_result = MultiReferenceVerifier(
                similarity_threshold=0.97,
                pass_consensus=0.67,
                review_consensus=0.34,
            ).verify(
                inspection_fingerprint,
                reference_set,
            )

            anomaly_detector = PatchMemoryAnomalyDetector(
                backbone=self.backbone,
                anomaly_threshold=0.25,
            )
            anomaly_detector.fit(variants)
            anomaly_result = anomaly_detector.predict(inspection_crop)

            fine = FineGrainedInspector(
                fingerprinter=self.fingerprinter,
                anomaly_detector=anomaly_detector,
                anomaly_scale=0.20,
            ).inspect(
                reference_crop,
                inspection_crop,
            )

            component_verification = verification_by_id[component.component_id]

            signals.append(
                ComponentSignals(
                    component_id=component.component_id,
                    component_type=component.component_type,
                    change_score=float(
                        np.clip(component_verification.overlap_ratio, 0.0, 1.0)
                    ),
                    fingerprint_similarity=float(
                        np.clip(fine.fingerprint_similarity, 0.0, 1.0)
                    ),
                    anomaly_score=float(np.clip(anomaly_result.score, 0.0, 1.0)),
                    edge_difference=float(np.clip(fine.edge_difference, 0.0, 1.0)),
                    texture_difference=float(
                        np.clip(fine.texture_difference, 0.0, 1.0)
                    ),
                    reference_consensus=float(
                        np.clip(reference_result.match.consensus_ratio, 0.0, 1.0)
                    ),
                )
            )

        diagnostics = PipelineDiagnostics(
            registration_inlier_ratio=float(registration.inlier_ratio),
            change_area_ratio=float(change.changed_area_ratio),
            changed_region_count=len(change.regions),
            detection_count=len(detections),
            component_count=len(registry.components),
        )

        return signals, diagnostics

    @staticmethod
    def _crop(image: np.ndarray, box) -> np.ndarray:
        height, width = image.shape[:2]
        x1 = max(0, min(int(box.x), width - 1))
        y1 = max(0, min(int(box.y), height - 1))
        x2 = max(x1 + 1, min(int(box.x2), width))
        y2 = max(y1 + 1, min(int(box.y2), height))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("component crop is empty")
        return crop

    @staticmethod
    def _normal_variants(reference_crop: np.ndarray) -> list[np.ndarray]:
        return [
            reference_crop.copy(),
            cv2.convertScaleAbs(reference_crop, alpha=0.98, beta=2),
            cv2.convertScaleAbs(reference_crop, alpha=1.02, beta=0),
        ]
