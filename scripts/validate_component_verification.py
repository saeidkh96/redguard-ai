from pathlib import Path

import cv2
import numpy as np

from redguard.imaging.change_detection import ChangeDetector
from redguard.imaging.registration import ImageRegistrationEngine
from redguard.inspection.component_verifier import (
    ComponentStatus,
    ComponentVerifier,
)
from redguard.models.component import (
    BoundingBox,
    ComponentDefinition,
)


ARTIFACT_DIR = Path("artifacts/component_verification")


def build_reference() -> np.ndarray:
    image = np.full(
        (600, 900, 3),
        25,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (60, 60),
        (840, 540),
        (65, 115, 65),
        -1,
    )

    cv2.rectangle(
        image,
        (330, 200),
        (570, 390),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        image,
        "RG-CPU-A1",
        (370, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    components = [
        ((130, 130), (190, 165), "Q14"),
        ((670, 120), (760, 160), "R27"),
        ((120, 430), (185, 490), "C08"),
        ((680, 420), (755, 485), "U03"),
    ]

    for top_left, bottom_right, label in components:
        cv2.rectangle(
            image,
            top_left,
            bottom_right,
            (180, 180, 180),
            -1,
        )

        cv2.putText(
            image,
            label,
            (
                top_left[0],
                top_left[1] - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    traces = [
        ((190, 150), (330, 230)),
        ((570, 250), (680, 150)),
        ((180, 450), (340, 350)),
        ((570, 350), (690, 450)),
    ]

    for start, end in traces:
        cv2.line(
            image,
            start,
            end,
            (210, 210, 210),
            3,
        )

    for point in (
        (95, 95),
        (805, 95),
        (95, 505),
        (805, 505),
    ):
        cv2.circle(
            image,
            point,
            14,
            (5, 5, 5),
            -1,
        )

    return image


def build_component_registry() -> list[ComponentDefinition]:
    return [
        ComponentDefinition(
            component_id="Q14",
            component_type="transistor",
            bounding_box=BoundingBox(
                x=130,
                y=130,
                width=60,
                height=35,
            ),
        ),
        ComponentDefinition(
            component_id="R27",
            component_type="resistor",
            bounding_box=BoundingBox(
                x=670,
                y=120,
                width=90,
                height=40,
            ),
        ),
        ComponentDefinition(
            component_id="C08",
            component_type="capacitor",
            bounding_box=BoundingBox(
                x=120,
                y=430,
                width=65,
                height=60,
            ),
        ),
        ComponentDefinition(
            component_id="U03",
            component_type="integrated_circuit",
            bounding_box=BoundingBox(
                x=680,
                y=420,
                width=75,
                height=65,
            ),
        ),
    ]


def create_changed_board(
    reference: np.ndarray,
) -> np.ndarray:
    changed = reference.copy()

    # Physically alter Q14 only.
    cv2.rectangle(
        changed,
        (138, 137),
        (182, 158),
        (35, 35, 35),
        -1,
    )

    cv2.line(
        changed,
        (143, 142),
        (177, 153),
        (245, 245, 245),
        2,
    )

    return changed


def transform(
    image: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        3.5,
        1.0,
    )

    matrix[0, 2] += 15
    matrix[1, 2] -= 10

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
    )


def draw_component_results(
    image: np.ndarray,
    registry: list[ComponentDefinition],
    verification_result,
) -> np.ndarray:
    output = image.copy()

    results_by_id = {
        item.component_id: item
        for item in verification_result.components
    }

    for component in registry:
        box = component.bounding_box
        result = results_by_id[component.component_id]

        if result.status == ComponentStatus.CHANGED:
            color = (0, 0, 255)
            label = f"{component.component_id}: CHANGED"
        else:
            color = (0, 255, 0)
            label = f"{component.component_id}: OK"

        cv2.rectangle(
            output,
            (box.x, box.y),
            (box.x2, box.y2),
            color,
            2,
        )

        cv2.putText(
            output,
            label,
            (box.x, max(20, box.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def main() -> int:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = build_reference()
    registry = build_component_registry()

    changed_board = create_changed_board(
        reference
    )

    inspection = transform(
        changed_board
    )

    registration_result = (
        ImageRegistrationEngine().register(
            reference,
            inspection,
        )
    )

    change_result = ChangeDetector(
        threshold=25,
        ssim_threshold=20,
        min_region_area=80,
        morphology_kernel_size=3,
        comparison_blur_size=5,
        border_margin=12,
    ).detect(
        reference,
        registration_result.aligned_image,
        valid_mask=registration_result.valid_mask,
    )

    verification_result = ComponentVerifier(
        min_overlap_ratio=0.05
    ).verify(
        registry,
        change_result.regions,
    )

    annotated = draw_component_results(
        registration_result.aligned_image,
        registry,
        verification_result,
    )

    cv2.imwrite(
        str(ARTIFACT_DIR / "reference.png"),
        reference,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "inspection_changed_transformed.png"
        ),
        inspection,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "inspection_aligned.png"
        ),
        registration_result.aligned_image,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "change_mask.png"
        ),
        change_result.binary_mask,
    )

    cv2.imwrite(
        str(
            ARTIFACT_DIR
            / "component_verification.png"
        ),
        annotated,
    )

    print(
        "RedGuard AI Component Verification Validation"
    )
    print("=" * 46)

    print(
        f"Registration inlier ratio: "
        f"{registration_result.inlier_ratio:.3f}"
    )

    print(
        f"Change regions:             "
        f"{len(change_result.regions)}"
    )

    print(
        f"Changed area ratio:         "
        f"{change_result.changed_area_ratio:.4%}"
    )

    print()

    for component in verification_result.components:
        print(
            f"{component.component_id:<4} "
            f"{component.component_type:<20} "
            f"{component.status.value.upper():<10} "
            f"overlap={component.overlap_ratio:.2%} "
            f"confidence={component.confidence:.3f}"
        )

    print()

    changed_ids = {
        component.component_id
        for component
        in verification_result.changed_components
    }

    unchanged_ids = {
        component.component_id
        for component
        in verification_result.unchanged_components
    }

    expected_changed = {"Q14"}
    expected_unchanged = {
        "R27",
        "C08",
        "U03",
    }

    if (
        changed_ids == expected_changed
        and unchanged_ids == expected_unchanged
        and registration_result.inlier_ratio >= 0.30
        and len(change_result.regions) <= 4
    ):
        print(
            "[PASS] Inspection image registered"
        )
        print(
            "[PASS] Visual change localized"
        )
        print(
            "[PASS] Change mapped to component Q14"
        )
        print(
            "[PASS] Unaffected components preserved"
        )
        print(
            "[PASS] Component verification artifacts generated"
        )
        print()
        print(
            "REDGUARD v0.1.0 "
            "COMPONENT VERIFICATION: PASS"
        )
        return 0

    print(
        "[FAIL] Component verification validation failed"
    )
    print()
    print(
        "REDGUARD v0.1.0 "
        "COMPONENT VERIFICATION: FAIL"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
