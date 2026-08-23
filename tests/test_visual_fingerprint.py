import cv2
import numpy as np
import pytest

from redguard.features.backbone import (
    VisionBackbone,
)
from redguard.features.fingerprint import (
    ComponentFingerprinter,
)


@pytest.fixture(scope="module")
def backbone():
    return VisionBackbone(
        pretrained=False
    )


def build_component():
    image = np.full(
        (160, 160, 3),
        40,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (35, 45),
        (125, 115),
        (180, 180, 180),
        -1,
    )

    cv2.putText(
        image,
        "Q14",
        (55, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
    )

    return image


def test_global_embedding_dimension(backbone):
    embedding = backbone.global_embedding(
        build_component()
    )

    assert embedding.shape == (512,)


def test_patch_embedding_shape(backbone):
    patches, shape = (
        backbone.patch_embeddings(
            build_component()
        )
    )

    assert patches.ndim == 2
    assert patches.shape[1] == 256
    assert shape[0] * shape[1] == patches.shape[0]


def test_identical_fingerprint_similarity(backbone):
    image = build_component()

    comparison = ComponentFingerprinter(
        backbone=backbone
    ).compare(
        image,
        image.copy(),
    )

    assert comparison.similarity > 0.999
    assert comparison.same_instance_candidate


def test_invalid_threshold():
    with pytest.raises(ValueError):
        ComponentFingerprinter(
            backbone=VisionBackbone(pretrained=False),
            similarity_threshold=1.5,
        )
