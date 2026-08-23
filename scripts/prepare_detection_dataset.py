from pathlib import Path
import random

import cv2
import numpy as np


DATA_ROOT = Path("data/detection")

CLASS_IDS = {
    "transistor": 0,
    "resistor": 1,
    "capacitor": 2,
    "integrated_circuit": 3,
}


def clear_generated_data() -> None:
    for split in (
        "train",
        "val",
        "test",
    ):
        image_dir = (
            DATA_ROOT
            / "images"
            / split
        )

        label_dir = (
            DATA_ROOT
            / "labels"
            / split
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for path in image_dir.glob(
            "synthetic_*"
        ):
            path.unlink()

        for path in label_dir.glob(
            "synthetic_*"
        ):
            path.unlink()


def yolo_box(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    return (
        (x + width / 2)
        / image_width,
        (y + height / 2)
        / image_height,
        width / image_width,
        height / image_height,
    )


def draw_component(
    image: np.ndarray,
    component_type: str,
    x: int,
    y: int,
) -> tuple[int, int]:
    if component_type == "transistor":
        width, height = 60, 48

    elif component_type == "resistor":
        width, height = 120, 30

    elif component_type == "capacitor":
        width, height = 38, 75

    elif (
        component_type
        == "integrated_circuit"
    ):
        width, height = 115, 85

    else:
        raise ValueError(
            f"Unknown component type: "
            f"{component_type}"
        )

    cv2.rectangle(
        image,
        (x, y),
        (
            x + width,
            y + height,
        ),
        (220, 220, 220),
        -1,
    )

    cv2.rectangle(
        image,
        (x, y),
        (
            x + width,
            y + height,
        ),
        (160, 160, 160),
        2,
    )

    return width, height


def generate_sample(
    split: str,
    index: int,
    rng: random.Random,
) -> None:
    height = 480
    width = 640

    image = np.full(
        (height, width, 3),
        28,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (25, 25),
        (615, 455),
        (55, 105, 55),
        -1,
    )

    component_types = list(
        CLASS_IDS
    )

    rng.shuffle(
        component_types
    )

    positions = [
        (70, 80),
        (300, 75),
        (90, 290),
        (350, 275),
    ]

    annotations = []

    for component_type, (
        base_x,
        base_y,
    ) in zip(
        component_types,
        positions,
        strict=True,
    ):
        jitter_x = rng.randint(
            -12,
            12,
        )

        jitter_y = rng.randint(
            -12,
            12,
        )

        x = base_x + jitter_x
        y = base_y + jitter_y

        component_width, component_height = (
            draw_component(
                image,
                component_type,
                x,
                y,
            )
        )

        (
            x_center,
            y_center,
            box_width,
            box_height,
        ) = yolo_box(
            x,
            y,
            component_width,
            component_height,
            width,
            height,
        )

        annotations.append(
            (
                CLASS_IDS[
                    component_type
                ],
                x_center,
                y_center,
                box_width,
                box_height,
            )
        )

    noise = rng.randint(
        0,
        10,
    )

    if noise:
        perturbation = np.random.default_rng(
            index
            + {
                "train": 1000,
                "val": 2000,
                "test": 3000,
            }[split]
        ).integers(
            0,
            noise + 1,
            size=image.shape,
            dtype=np.uint8,
        )

        image = cv2.add(
            image,
            perturbation,
        )

    stem = (
        f"synthetic_{split}_"
        f"{index:03d}"
    )

    image_path = (
        DATA_ROOT
        / "images"
        / split
        / f"{stem}.png"
    )

    label_path = (
        DATA_ROOT
        / "labels"
        / split
        / f"{stem}.txt"
    )

    cv2.imwrite(
        str(image_path),
        image,
    )

    label_lines = [
        (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )
        for (
            class_id,
            x_center,
            y_center,
            box_width,
            box_height,
        ) in annotations
    ]

    label_path.write_text(
        "\n".join(label_lines)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    clear_generated_data()

    counts = {
        "train": 24,
        "val": 8,
        "test": 8,
    }

    rng = random.Random(
        42
    )

    for split, count in (
        counts.items()
    ):
        for index in range(
            count
        ):
            generate_sample(
                split,
                index,
                rng,
            )

    print(
        "RedGuard AI Detection "
        "Dataset Preparation"
    )
    print("=" * 41)

    for split, count in (
        counts.items()
    ):
        print(
            f"{split:<5}: "
            f"{count} images"
        )

    print()
    print(
        "Synthetic YOLO-format "
        "dataset generated."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
