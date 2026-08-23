from pathlib import Path

import torch
from ultralytics import YOLO


DATA_CONFIG = Path("configs/components.yaml")


def main() -> int:
    print("RedGuard AI Component Detector Training")
    print("=" * 42)

    if not DATA_CONFIG.exists():
        print(
            f"[FAIL] Dataset config not found: "
            f"{DATA_CONFIG}"
        )
        return 1

    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Training device: {device}")
    print()

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(DATA_CONFIG),
        epochs=20,
        imgsz=640,
        batch=4,
        device=device,
        workers=0,
        project="redguard_detection",
        name="v020_baseline",
        exist_ok=True,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"

    print()
    print(f"Training output: {save_dir}")

    if not best_model.exists():
        print(
            f"[FAIL] best.pt not found: "
            f"{best_model}"
        )
        return 1

    print("[PASS] YOLO training completed")
    print(f"[PASS] Model artifact: {best_model}")
    print()
    print(
        "REDGUARD v0.2.0 "
        "YOLO TRAINING: PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
