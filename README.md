# RedGuard AI

**Precision Visual Inspection & Physical Component Verification**

RedGuard AI is a computer vision platform for detecting, localizing, identifying, and verifying physical changes in components and visual systems.

The core idea is simple:

> Software state may remain unchanged while the physical system has changed.

RedGuard AI combines image registration, visual change detection, automatic component detection, visual fingerprinting, anomaly detection, and fine-grained inspection to reason about the physical state of components.

---

## Current Version

**v0.5.0 — Fine-Grained Component Inspection**

---

## Current Capabilities

RedGuard AI currently supports:

- Image loading and validation
- Image preprocessing and normalization
- Geometric image registration
- Registration artifact suppression
- Pixel-level visual change detection
- Changed-region localization
- Component-level change mapping
- Automatic component detection with YOLO
- Automatic component registry generation
- Stable component identifiers
- Component visual fingerprinting
- Same-instance-like visual comparison
- Patch-memory anomaly detection
- Spatial anomaly localization maps
- Edge-level difference analysis
- Texture-level difference analysis
- Multi-signal inspection risk scoring
- PASS / REVIEW / FAIL inspection decisions

---

## Inspection Pipeline

```text
Reference Image                 Inspection Image
      |                               |
      +---------------+---------------+
                      |
              Image Preprocessing
                      |
              Image Registration
                      |
              Change Detection
                      |
          Automatic Component Detection
                      |
             Component Registry
                      |
        +-------------+-------------+
        |             |             |
     Visual         Patch       Fine-Grained
   Fingerprint     Anomaly       Inspection
   Comparison      Detection
        |             |             |
        +-------------+-------------+
                      |
                 Risk Fusion
                      |
              PASS / REVIEW / FAIL
```

---

## Automatic Component Detection

RedGuard AI includes a YOLO-based component detector behind a common detector abstraction.

The current prototype supports four component classes:

- Transistor
- Resistor
- Capacitor
- Integrated circuit

The detector produces:

- Component class
- Bounding box
- Detection confidence
- Automatic component registry entries
- Stable component identifiers

Example automatically generated identifiers:

```text
resistor_001
transistor_001
capacitor_001
integrated_circuit_001
```

### Synthetic Detection Validation

Current YOLO validation on the RedGuard synthetic test dataset produced:

| Metric | Result |
|---|---:|
| Precision | 0.9367 |
| Recall | 0.9262 |
| mAP50 | 0.9950 |
| mAP50-95 | 0.9313 |

All four component classes were represented during validation.

> These metrics are measured on the current synthetic RedGuard validation dataset. They must not be interpreted as production or real-world PCB inspection performance.

---

## Visual Fingerprinting

**v0.3.0** introduced deep visual fingerprints for individual components.

A pretrained ResNet18 feature extractor converts component images into normalized visual embeddings.

The resulting fingerprints can be compared using cosine similarity to distinguish between:

- Same-instance-like observations
- Visually altered observations
- Replacement-like observations

Experimental validation:

```text
Decision threshold:             0.9700

Same-instance-like similarity:  0.9998
Same-instance decision:         True

Replacement similarity:         0.9394
Replacement same-instance:      False
```

The validation demonstrates that the current experimental threshold accepts the controlled same-instance-like sample while rejecting the controlled replacement-like sample.

This capability provides a foundation for physical component identity verification.

> Current fingerprint thresholds are experimental and validated on controlled examples. They are not yet calibrated for production identity verification.

---

## Patch-Memory Anomaly Detection

**v0.4.0** introduced local visual anomaly detection.

RedGuard builds a memory bank of deep patch embeddings extracted from known-normal reference images.

During inspection, query patches are compared with this normal feature memory. Patches that differ substantially from known-normal features receive higher anomaly scores.

Experimental validation:

```text
Decision threshold: 0.2500

Normal score:       0.1171
Normal anomalous:   False

Anomaly score:      0.9450
Anomaly detected:   True
```

The subsystem also generates spatial anomaly maps that can be used to localize unusual visual regions.

> Current anomaly thresholds and validation samples are experimental and are not yet calibrated against a real-world industrial defect dataset.

---

## Fine-Grained Inspection

**v0.5.0** combines multiple independent visual signals into a single component-level inspection result.

The inspection engine currently evaluates:

1. Deep visual fingerprint similarity
2. Patch-memory anomaly score
3. Edge differences
4. Texture differences

These signals are normalized and fused into a component-level risk score.

The resulting inspection decision is mapped to:

```text
PASS
REVIEW
FAIL
```

Experimental validation:

```text
Normal risk:          0.0007
Normal decision:      PASS

Altered fingerprint:  0.6836
Altered anomaly:      0.9886
Altered edge diff:    0.0088
Altered texture:      0.0091
Altered risk:         0.6239
Altered decision:     FAIL
```

This moves RedGuard beyond simple object detection.

The system can now ask both:

> What component is this?

and:

> Does this component still visually match its expected physical state?

---

## Image Registration

Inspection images are geometrically aligned with reference images before visual comparison.

This reduces false visual differences caused by camera movement or image misalignment.

Validated registration example:

```text
Matches total: 1410
Matches used:  649
Inliers:       556
Inlier ratio:  0.857

Error before:  12.780
Error after:   2.177
Improvement:   82.97%
```

---

## Visual Change Detection

After registration, RedGuard suppresses registration residuals and detects meaningful visual changes.

Validated example:

```text
Global similarity:   0.9573
Changed:             True
Changed regions:     1
Changed pixels:      1124
Changed area ratio:  0.2277%
```

The detected change was localized to a single meaningful region while false-positive area remained controlled.

---

## Component-Level Verification

Detected visual changes can be mapped to physical components.

Example validation:

```text
Q14  transistor          CHANGED
R27  resistor            UNCHANGED
C08  capacitor           UNCHANGED
U03  integrated_circuit  UNCHANGED
```

This allows RedGuard to reason about component state rather than only producing a global image-difference score.

---

## Detection-to-Verification Integration

Automatic detection is connected directly to the verification pipeline:

```text
Inspection Image
      |
      v
YOLO Component Detection
      |
      v
Automatic Component Registry
      |
      v
Stable Component IDs
      |
      v
Visual Change Mapping
      |
      v
Component Verification
```

Validated example:

```text
YOLO detections:          4
Generated registry size:  4
Detector source:          yolo11n-redguard

resistor_001             CHANGED
transistor_001           UNCHANGED
capacitor_001            UNCHANGED
integrated_circuit_001   UNCHANGED
```

Automatic component detection therefore removes the requirement to manually define every component bounding box before verification.

---

## Architecture

```text
                         RedGuard AI
                              |
        +---------------------+---------------------+
        |                     |                     |
      Imaging              Detection            Inspection
        |                     |                     |
 Preprocessing          Baseline Detector     Change Detection
 Registration           YOLO Detector         Component Verification
 Validation             Dataset Pipeline      Fine-Grained Inspection
                              |
                       Component Registry
                              |
                 +------------+------------+
                 |                         |
              Features                   Anomaly
                 |                         |
          ResNet18 Backbone          Patch Memory
          Visual Fingerprint         Anomaly Maps
                 |                         |
                 +------------+------------+
                              |
                         Risk Fusion
                              |
                     PASS / REVIEW / FAIL
```

---

## Core Modules

### Imaging

Responsible for:

- Image loading
- Image validation
- Preprocessing
- Normalization
- Geometric registration
- Registration quality estimation
- Visual change detection

### Detection

Responsible for:

- Baseline component detection
- YOLO component detection
- Detection dataset management
- Bounding-box generation
- Component classification
- Detection confidence
- Automatic registry generation

### Features

Responsible for:

- Shared deep feature extraction
- ResNet18 visual backbone
- Global component embeddings
- Patch embeddings
- Component visual fingerprints
- Fingerprint comparison

### Anomaly

Responsible for:

- Normal patch-memory construction
- Nearest-neighbour feature comparison
- Local anomaly scoring
- Spatial anomaly maps

### Inspection

Responsible for:

- Component-level verification
- Fine-grained inspection
- Edge comparison
- Texture comparison
- Multi-signal risk fusion
- PASS / REVIEW / FAIL decisions

---

## Validation

Current automated test suite:

```text
96 passed
```

Major validated flows include:

- Foundation Validation
- Image Preprocessing Validation
- Image Registration Validation
- Visual Change Detection Validation
- Component Verification Validation
- Automatic Component Detection Validation
- Detection Dataset Validation
- YOLO Detector Validation
- Detection-to-Verification Validation
- Visual Fingerprint Validation
- Patch Anomaly Validation
- Fine-Grained Inspection Validation

---

## Current Experimental Results

| Capability | Validation Result |
|---|---:|
| Registration improvement | 82.97% |
| YOLO Precision | 0.9367 |
| YOLO Recall | 0.9262 |
| YOLO mAP50 | 0.9950 |
| YOLO mAP50-95 | 0.9313 |
| Same-instance similarity | 0.9998 |
| Replacement-like similarity | 0.9394 |
| Normal anomaly score | 0.1171 |
| Altered anomaly score | 0.9450 |
| Normal inspection | PASS |
| Altered inspection | FAIL |
| Automated tests | 96 passed |

> Detection metrics currently come from the synthetic RedGuard dataset. Fingerprint, anomaly, and fine-grained inspection results come from controlled experimental validation scenarios. These numbers are development validation results, not production benchmarks.

---

## Version History

| Version | Milestone | Status |
|---|---|---|
| v0.0.1 | Project Foundation | Complete |
| v0.0.2 | Image Preprocessing | Complete |
| v0.0.3 | Image Registration | Complete |
| v0.0.4 | Baseline Change Detection | Complete |
| v0.1.0 | Component-Level Verification | Complete |
| v0.2.0 | Automatic Component Detection | Complete |
| v0.3.0 | Component Visual Fingerprinting | Complete |
| v0.4.0 | Patch-Memory Anomaly Detection | Complete |
| v0.5.0 | Fine-Grained Inspection | Complete |
| v0.6.0 | Multi-Reference Verification | Complete |
| v0.7.0 | Inspection Intelligence | Complete |
| v0.8.0 | Vision AI Reasoning | **Current** |
| v0.9.0 | Production Inspection System | Planned |
| v0.10.0 | Production Validation | Planned |
| v1.0.0 | RedGuard AI | Planned |

---

## Next Milestone

### v0.6.0 — Multi-Reference Verification

The next milestone will move RedGuard from single-reference comparison toward reference-set-based inspection.

Planned capabilities include:

- Multiple normal references per component
- Reference feature banks
- Best-reference matching
- Reference consensus
- Robustness to illumination variation
- Robustness to small viewpoint variation
- Reference-set fingerprint comparison
- Reference-set anomaly baselines
- Confidence-aware verification

Instead of asking:

```text
Does inspection image B match reference image A?
```

RedGuard will move toward:

```text
Is this inspection consistent with the known-normal
visual state of this physical component?
```

This is intended to make verification less dependent on one perfect reference image.

---

## Roadmap

```text
v0.0.x
Foundation / Preprocessing / Registration / Change Detection
        |
        v
v0.1.0
Component-Level Verification
        |
        v
v0.2.0
Automatic Component Detection
        |
        v
v0.3.0
Visual Fingerprinting
        |
        v
v0.4.0
Patch-Memory Anomaly Detection
        |
        v
v0.5.0
Fine-Grained Inspection
        |
        v
v0.6.0
Multi-Reference Verification
        |
        v
v0.7.0
Inspection Intelligence
        |
        v
v0.8.0
Vision AI Reasoning
        |
        v
v0.9.0
Production Inspection System
        |
        v
v0.10.0
Production Validation
        |
        v
v1.0.0
RedGuard AI
```

---

## Project Direction

RedGuard AI is being developed toward a physical-state verification system combining:

```text
Object Detection
       +
Image Registration
       +
Change Detection
       +
Visual Fingerprinting
       +
Anomaly Detection
       +
Fine-Grained Inspection
       +
Inspection Intelligence
```

The long-term goal is to detect physical changes that may not be visible from software state, identifiers, component metadata, or part numbers alone.

Potential inspection scenarios include:

- Component replacement
- Missing components
- Surface damage
- Cracks
- Burn marks
- Deformation
- Unexpected visual modifications
- Fine-grained physical-state changes
- Identity inconsistencies between reference and inspection observations

---

## Technology

Current RedGuard AI development includes:

- Python
- OpenCV
- NumPy
- PyTorch
- Torchvision
- Ultralytics YOLO
- scikit-image
- Pydantic
- PyYAML
- pytest

---

## Project Status

RedGuard AI is an experimental research and engineering project under active development.

Current validation primarily uses controlled and synthetic data.

The project does **not** currently claim:

- Production-grade defect detection accuracy
- Production-grade component identity verification
- Generalization across arbitrary PCB layouts
- Industrial threshold calibration
- Certified quality-control performance

Real-world datasets, robustness evaluation, threshold calibration, deployment testing, and production validation remain future milestones.

---

## License

See `LICENSE`.
