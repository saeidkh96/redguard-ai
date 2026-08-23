from pathlib import Path

import cv2
import numpy as np
import pytest

from redguard.detection.dataset import (
    DetectionDatasetConfig,
    YoloAnnotation,
    collect_split_samples,
    load_dataset_config,
    parse_yolo_label,
    summarize_split,
)


def test_valid_annotation():
    annotation = YoloAnnotation(
        class_id=1,
        x_center=0.5,
        y_center=0.5,
        width=0.2,
        height=0.3,
    )

    assert annotation.class_id == 1


def test_invalid_normalized_coordinate():
    with pytest.raises(ValueError):
        YoloAnnotation(
            class_id=0,
            x_center=1.2,
            y_center=0.5,
            width=0.2,
            height=0.2,
        )


def test_zero_width_is_rejected():
    with pytest.raises(ValueError):
        YoloAnnotation(
            class_id=0,
            x_center=0.5,
            y_center=0.5,
            width=0.0,
            height=0.2,
        )


def test_parse_valid_label(
    tmp_path: Path,
):
    label = tmp_path / "sample.txt"

    label.write_text(
        "1 0.5 0.5 0.2 0.3\n",
        encoding="utf-8",
    )

    annotations = parse_yolo_label(
        label,
        valid_class_ids={
            0,
            1,
            2,
            3,
        },
    )

    assert len(annotations) == 1
    assert annotations[0].class_id == 1


def test_unknown_class_is_rejected(
    tmp_path: Path,
):
    label = tmp_path / "sample.txt"

    label.write_text(
        "9 0.5 0.5 0.2 0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        parse_yolo_label(
            label,
            valid_class_ids={
                0,
                1,
                2,
                3,
            },
        )


def build_temp_dataset(
    tmp_path: Path,
) -> DetectionDatasetConfig:
    root = (
        tmp_path
        / "detection"
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        (
            root
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            root
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    return DetectionDatasetConfig(
        root=root,
        train_images=(
            root
            / "images"
            / "train"
        ),
        val_images=(
            root
            / "images"
            / "val"
        ),
        test_images=(
            root
            / "images"
            / "test"
        ),
        class_names={
            0: "transistor",
            1: "resistor",
            2: "capacitor",
            3: "integrated_circuit",
        },
    )


def test_collect_sample_pair(
    tmp_path: Path,
):
    config = build_temp_dataset(
        tmp_path
    )

    image_path = (
        config.train_images
        / "sample.png"
    )

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    assert cv2.imwrite(
        str(image_path),
        image,
    )

    label_path = (
        config.root
        / "labels"
        / "train"
        / "sample.txt"
    )

    label_path.write_text(
        "0 0.5 0.5 0.3 0.3\n",
        encoding="utf-8",
    )

    samples = collect_split_samples(
        config,
        "train",
    )

    assert len(samples) == 1
    assert len(
        samples[0].annotations
    ) == 1


def test_missing_label_is_rejected(
    tmp_path: Path,
):
    config = build_temp_dataset(
        tmp_path
    )

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    cv2.imwrite(
        str(
            config.train_images
            / "sample.png"
        ),
        image,
    )

    with pytest.raises(ValueError):
        collect_split_samples(
            config,
            "train",
        )


def test_split_summary_counts_classes(
    tmp_path: Path,
):
    config = build_temp_dataset(
        tmp_path
    )

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    cv2.imwrite(
        str(
            config.train_images
            / "sample.png"
        ),
        image,
    )

    (
        config.root
        / "labels"
        / "train"
        / "sample.txt"
    ).write_text(
        (
            "0 0.3 0.3 0.2 0.2\n"
            "1 0.7 0.7 0.2 0.2\n"
        ),
        encoding="utf-8",
    )

    summary = summarize_split(
        config,
        "train",
    )

    assert summary.images == 1
    assert summary.annotations == 2
    assert summary.class_counts[0] == 1
    assert summary.class_counts[1] == 1
