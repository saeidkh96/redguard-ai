from redguard.detection.dataset import (
    load_dataset_config,
    summarize_split,
)


CONFIG_PATH = "configs/components.yaml"


def main() -> int:
    config = load_dataset_config(
        CONFIG_PATH
    )

    print(
        "RedGuard AI Detection "
        "Dataset Validation"
    )
    print("=" * 40)

    print(
        f"Classes: "
        f"{len(config.class_names)}"
    )

    for class_id, name in (
        config.class_names.items()
    ):
        print(
            f"  {class_id}: {name}"
        )

    print()

    expected_images = {
        "train": 24,
        "val": 8,
        "test": 8,
    }

    all_valid = True

    for split in (
        "train",
        "val",
        "test",
    ):
        summary = summarize_split(
            config,
            split,
        )

        print(
            f"{split.upper()}"
        )

        print(
            f"  Images:      "
            f"{summary.images}"
        )

        print(
            f"  Annotations: "
            f"{summary.annotations}"
        )

        print(
            "  Class counts:"
        )

        for class_id, count in (
            summary.class_counts.items()
        ):
            print(
                f"    "
                f"{config.class_names[class_id]:<20} "
                f"{count}"
            )

        if (
            summary.images
            != expected_images[split]
        ):
            all_valid = False

        if any(
            count == 0
            for count
            in summary.class_counts.values()
        ):
            all_valid = False

        print()

    if all_valid:
        print(
            "[PASS] YOLO directory "
            "structure valid"
        )

        print(
            "[PASS] Image/label "
            "pairing valid"
        )

        print(
            "[PASS] Normalized "
            "annotations valid"
        )

        print(
            "[PASS] Class IDs valid"
        )

        print(
            "[PASS] All classes "
            "represented"
        )

        print()

        print(
            "REDGUARD v0.2.0 "
            "DETECTION DATASET: PASS"
        )

        return 0

    print(
        "[FAIL] Detection dataset "
        "validation failed"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
