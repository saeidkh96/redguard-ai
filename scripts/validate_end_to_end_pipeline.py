from __future__ import annotations

from pathlib import Path
import shutil

import cv2
import numpy as np

from redguard.orchestration import (
    ImageSignalExtractor,
    ProductionInspectionPipeline,
)
from redguard.persistence import InspectionRepository
from redguard.services import ArtifactService, InspectionService


ROOT = Path("artifacts/v010_end_to_end")


def build_reference() -> np.ndarray:
    image = np.full((600, 900, 3), 30, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (850, 550), (70, 105, 70), -1)

    # ORB-friendly sub-threshold visual structure.
    for x in range(90, 830, 80):
        cv2.circle(image, (x, 90), 9, (125, 125, 125), -1)
        cv2.line(image, (x, 500), (x + 35, 465), (125, 125, 125), 2)

    cv2.putText(
        image,
        "REDGUARD BOARD A-01",
        (250, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (125, 125, 125),
        2,
        cv2.LINE_AA,
    )

    # Bright components detected by the deterministic baseline detector.
    cv2.rectangle(image, (120, 150), (260, 185), (225, 225, 225), -1)  # resistor
    cv2.rectangle(image, (610, 155), (670, 205), (225, 225, 225), -1)  # transistor

    return image


def build_inspection(reference: np.ndarray) -> np.ndarray:
    altered = reference.copy()

    # Strong localized physical-state change on the resistor-like component.
    cv2.rectangle(altered, (150, 155), (230, 180), (25, 25, 25), -1)
    cv2.line(altered, (155, 158), (225, 177), (245, 245, 245), 3)

    height, width = altered.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 2.0, 1.0)
    matrix[0, 2] += 9
    matrix[1, 2] -= 6

    return cv2.warpAffine(altered, matrix, (width, height))


def main() -> int:
    # Validation must be repeatable. Remove state from previous validation runs.
    if ROOT.exists():
        shutil.rmtree(ROOT)

    ROOT.mkdir(parents=True, exist_ok=True)

    reference = build_reference()
    inspection = build_inspection(reference)

    cv2.imwrite(str(ROOT / "reference.png"), reference)
    cv2.imwrite(str(ROOT / "inspection.png"), inspection)

    repository = InspectionRepository(ROOT / "history.json")
    artifacts = ArtifactService(ROOT / "inspections")
    service = InspectionService(repository, artifacts)

    pipeline = ProductionInspectionPipeline(
        service=service,
        signal_extractor=ImageSignalExtractor(pretrained_backbone=False),
    )

    result = pipeline.run(reference, inspection)

    print("RedGuard AI End-to-End Production Pipeline Validation")
    print("=" * 55)
    print(f"Registration inlier ratio: {result.diagnostics.registration_inlier_ratio:.3f}")
    print(f"Changed area ratio:        {result.diagnostics.change_area_ratio:.4%}")
    print(f"Changed regions:           {result.diagnostics.changed_region_count}")
    print(f"Detected components:       {result.diagnostics.detection_count}")
    print(f"Registry components:       {result.diagnostics.component_count}")
    print(f"Quality gate:              {'PASS' if result.quality_passed else 'FAIL'}")
    print()

    if result.quality_messages:
        for message in result.quality_messages:
            print(f"Quality: {message}")

    print("Persisted decisions")
    print("-------------------")
    for component_id, decision in result.decisions.items():
        print(f"{component_id:<28} {decision}")

    history = service.history()
    print()
    print(f"Persisted inspections:     {len(history)}")

    checks = [
        (result.quality_passed, "Visual quality gate passed"),
        (result.diagnostics.registration_inlier_ratio >= 0.30, "Registration validated"),
        (result.diagnostics.detection_count >= 2, "Components detected"),
        (len(result.inspection_ids) == result.diagnostics.component_count, "All components persisted"),
        (len(history) == len(result.inspection_ids), "Inspection history matches pipeline output"),
        (all(record.explanation for record in history), "Safe reasoning explanations persisted"),
        (all(record.artifacts for record in history), "Structured report artifacts persisted"),
        (any(record.decision in {"REVIEW", "FAIL"} for record in history), "Altered physical state surfaced"),
    ]

    print()
    print("Validation")
    print("----------")

    failed = False
    for passed, message in checks:
        state = "PASS" if passed else "FAIL"
        print(f"[{state}] {message}")
        failed = failed or not passed

    print()
    if failed:
        print("REDGUARD v0.10.0 END-TO-END PIPELINE: FAIL")
        return 1

    print("REDGUARD v0.10.0 END-TO-END PIPELINE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
