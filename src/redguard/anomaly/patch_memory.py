from dataclasses import dataclass

import cv2
import numpy as np
import torch

from redguard.features.backbone import (
    VisionBackbone,
)


@dataclass(frozen=True)
class AnomalyResult:
    score: float
    is_anomalous: bool
    anomaly_map: np.ndarray


class PatchMemoryAnomalyDetector:
    """
    Patch-memory anomaly detector inspired by PatchCore.

    The memory bank contains patch embeddings from normal
    component images. Test patches are scored using nearest
    neighbour distance to the normal feature bank.
    """

    def __init__(
        self,
        backbone: VisionBackbone | None = None,
        anomaly_threshold: float = 0.25,
        max_memory_patches: int = 2048,
        percentile: float = 95.0,
    ) -> None:
        if anomaly_threshold <= 0:
            raise ValueError(
                "anomaly_threshold must be positive."
            )

        if max_memory_patches <= 0:
            raise ValueError(
                "max_memory_patches must be positive."
            )

        if not 0 < percentile <= 100:
            raise ValueError(
                "percentile must be in (0,100]."
            )

        self.backbone = (
            backbone
            or VisionBackbone()
        )

        self.anomaly_threshold = (
            anomaly_threshold
        )

        self.max_memory_patches = (
            max_memory_patches
        )

        self.percentile = percentile
        self.memory_bank: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.memory_bank is not None

    def fit(
        self,
        normal_images: list[np.ndarray],
    ) -> None:
        if not normal_images:
            raise ValueError(
                "normal_images cannot be empty."
            )

        patch_sets = []

        for image in normal_images:
            patches, _ = (
                self.backbone.patch_embeddings(
                    image
                )
            )

            patch_sets.append(
                patches
            )

        memory = np.concatenate(
            patch_sets,
            axis=0,
        )

        if (
            len(memory)
            > self.max_memory_patches
        ):
            rng = np.random.default_rng(
                42
            )

            indices = rng.choice(
                len(memory),
                self.max_memory_patches,
                replace=False,
            )

            memory = memory[
                indices
            ]

        self.memory_bank = (
            memory.astype(np.float32)
        )

    def predict(
        self,
        image: np.ndarray,
    ) -> AnomalyResult:
        if self.memory_bank is None:
            raise RuntimeError(
                "Anomaly detector must be fitted first."
            )

        patches, patch_shape = (
            self.backbone.patch_embeddings(
                image
            )
        )

        query = torch.from_numpy(
            patches
        )

        memory = torch.from_numpy(
            self.memory_bank
        )

        distances = torch.cdist(
            query,
            memory,
            p=2,
        )

        nearest = (
            distances
            .min(dim=1)
            .values
            .numpy()
        )

        score = float(
            np.percentile(
                nearest,
                self.percentile,
            )
        )

        patch_map = nearest.reshape(
            patch_shape
        )

        anomaly_map = cv2.resize(
            patch_map,
            (
                image.shape[1],
                image.shape[0],
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        return AnomalyResult(
            score=score,
            is_anomalous=(
                score
                >= self.anomaly_threshold
            ),
            anomaly_map=anomaly_map.astype(
                np.float32
            ),
        )
