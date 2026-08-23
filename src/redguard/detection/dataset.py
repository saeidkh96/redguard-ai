from dataclasses import dataclass
from pathlib import Path

import yaml


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


@dataclass(frozen=True)
class YoloAnnotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError(
                "class_id cannot be negative."
            )

        for name, value in (
            ("x_center", self.x_center),
            ("y_center", self.y_center),
            ("width", self.width),
            ("height", self.height),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be normalized between 0 and 1."
                )

        if self.width <= 0.0:
            raise ValueError(
                "width must be greater than zero."
            )

        if self.height <= 0.0:
            raise ValueError(
                "height must be greater than zero."
            )


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    label_path: Path
    annotations: tuple[YoloAnnotation, ...]


@dataclass(frozen=True)
class DetectionDatasetConfig:
    root: Path
    train_images: Path
    val_images: Path
    test_images: Path
    class_names: dict[int, str]


@dataclass(frozen=True)
class DatasetSplitSummary:
    split: str
    images: int
    annotations: int
    class_counts: dict[int, int]


def load_dataset_config(
    config_path: str | Path,
) -> DetectionDatasetConfig:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset config does not exist: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(
            "Dataset configuration must be a mapping."
        )

    required = {
        "path",
        "train",
        "val",
        "test",
        "names",
    }

    missing = required - raw.keys()

    if missing:
        raise ValueError(
            f"Dataset config missing keys: "
            f"{sorted(missing)}"
        )

    names_raw = raw["names"]

    if not isinstance(names_raw, dict):
        raise ValueError(
            "names must be a mapping."
        )

    class_names = {
        int(class_id): str(name)
        for class_id, name in names_raw.items()
    }

    if not class_names:
        raise ValueError(
            "At least one detection class is required."
        )

    expected_ids = set(
        range(len(class_names))
    )

    if set(class_names) != expected_ids:
        raise ValueError(
            "Class IDs must be contiguous starting at 0."
        )

    root = Path(raw["path"])

    return DetectionDatasetConfig(
        root=root,
        train_images=root / raw["train"],
        val_images=root / raw["val"],
        test_images=root / raw["test"],
        class_names=class_names,
    )


def parse_yolo_label(
    label_path: str | Path,
    valid_class_ids: set[int] | None = None,
) -> tuple[YoloAnnotation, ...]:
    label_path = Path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file does not exist: {label_path}"
        )

    annotations: list[YoloAnnotation] = []

    for line_number, raw_line in enumerate(
        label_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO annotation at "
                f"{label_path}:{line_number}. "
                f"Expected 5 values."
            )

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric annotation at "
                f"{label_path}:{line_number}."
            ) from exc

        if (
            valid_class_ids is not None
            and class_id not in valid_class_ids
        ):
            raise ValueError(
                f"Unknown class ID {class_id} at "
                f"{label_path}:{line_number}."
            )

        annotations.append(
            YoloAnnotation(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return tuple(annotations)


def collect_split_samples(
    config: DetectionDatasetConfig,
    split: str,
) -> tuple[DetectionSample, ...]:
    image_dir = _image_dir_for_split(
        config,
        split,
    )

    label_dir = (
        config.root
        / "labels"
        / split
    )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: "
            f"{image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory does not exist: "
            f"{label_dir}"
        )

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )
    )

    valid_class_ids = set(
        config.class_names
    )

    samples: list[DetectionSample] = []

    for image_path in image_paths:
        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise ValueError(
                f"Missing label file for image: "
                f"{image_path.name}"
            )

        annotations = parse_yolo_label(
            label_path,
            valid_class_ids=valid_class_ids,
        )

        samples.append(
            DetectionSample(
                image_path=image_path,
                label_path=label_path,
                annotations=annotations,
            )
        )

    label_stems = {
        path.stem
        for path in label_dir.glob("*.txt")
    }

    image_stems = {
        path.stem
        for path in image_paths
    }

    orphan_labels = (
        label_stems - image_stems
    )

    if orphan_labels:
        raise ValueError(
            f"Label files without images in "
            f"{split}: "
            f"{sorted(orphan_labels)}"
        )

    return tuple(samples)


def summarize_split(
    config: DetectionDatasetConfig,
    split: str,
) -> DatasetSplitSummary:
    samples = collect_split_samples(
        config,
        split,
    )

    class_counts = {
        class_id: 0
        for class_id in config.class_names
    }

    annotation_count = 0

    for sample in samples:
        for annotation in sample.annotations:
            annotation_count += 1
            class_counts[
                annotation.class_id
            ] += 1

    return DatasetSplitSummary(
        split=split,
        images=len(samples),
        annotations=annotation_count,
        class_counts=class_counts,
    )


def _image_dir_for_split(
    config: DetectionDatasetConfig,
    split: str,
) -> Path:
    mapping = {
        "train": config.train_images,
        "val": config.val_images,
        "test": config.test_images,
    }

    if split not in mapping:
        raise ValueError(
            f"Unknown dataset split: {split}"
        )

    return mapping[split]
