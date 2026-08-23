import cv2
import numpy as np

from redguard.anomaly.patch_memory import PatchMemoryAnomalyDetector
from redguard.features.backbone import VisionBackbone
from redguard.features.fingerprint import ComponentFingerprinter
from redguard.inspection.fine_grained import FineGrainedInspector


def build_component():
    image = np.full(
        (224,224,3),
        35,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (55,65),
        (170,160),
        (175,175,175),
        -1,
    )

    cv2.putText(
        image,
        "Q14 A7",
        (70,115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20,20,20),
        2,
    )

    return image


def main():
    reference = build_component()

    normal_variants = []

    for delta in range(5):
        normal_variants.append(
            cv2.convertScaleAbs(
                reference,
                alpha=1.0,
                beta=delta,
            )
        )

    altered = reference.copy()

    cv2.line(
        altered,
        (70,75),
        (155,150),
        (255,255,255),
        9,
    )

    cv2.circle(
        altered,
        (120,115),
        18,
        (30,30,30),
        -1,
    )

    backbone = VisionBackbone(
        pretrained=True
    )

    anomaly = PatchMemoryAnomalyDetector(
        backbone=backbone,
        anomaly_threshold=0.25,
    )

    anomaly.fit(
        normal_variants
    )

    inspector = FineGrainedInspector(
        fingerprinter=ComponentFingerprinter(
            backbone=backbone,
            similarity_threshold=0.97,
        ),
        anomaly_detector=anomaly,
    )

    normal_result = inspector.inspect(
        reference,
        reference.copy(),
    )

    altered_result = inspector.inspect(
        reference,
        altered,
    )

    print("RedGuard AI Fine-Grained Inspection Validation")
    print("=" * 47)

    print(f"Normal risk:        {normal_result.risk_score:.4f}")
    print(f"Normal decision:    {normal_result.decision.value.upper()}")
    print()
    print(f"Altered fingerprint:{altered_result.fingerprint_similarity:.4f}")
    print(f"Altered anomaly:    {altered_result.anomaly_score:.4f}")
    print(f"Altered edge diff:  {altered_result.edge_difference:.4f}")
    print(f"Altered texture:    {altered_result.texture_difference:.4f}")
    print(f"Altered risk:       {altered_result.risk_score:.4f}")
    print(f"Altered decision:   {altered_result.decision.value.upper()}")

    if (
        normal_result.risk_score
        < altered_result.risk_score
        and normal_result.decision.value == "pass"
        and altered_result.decision.value in {"review", "fail"}
    ):
        print()
        print("[PASS] Fine-grained identity signal generated")
        print("[PASS] Local anomaly signal generated")
        print("[PASS] Edge/texture changes measured")
        print("[PASS] Multi-signal risk score generated")
        print("[PASS] PASS/REVIEW/FAIL decision generated")
        print()
        print("REDGUARD v0.5.0 FINE-GRAINED INSPECTION: PASS")
        return 0

    print("[FAIL] Fine-grained validation failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
