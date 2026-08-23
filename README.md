# RedGuard AI

**Precision Visual Inspection & Physical Component Verification**

RedGuard AI is a computer vision platform for detecting, localizing, and verifying physical changes in components and visual systems.

The core idea is simple: software state may remain unchanged while the physical system has changed.

RedGuard AI aligns reference and inspection images, suppresses camera and registration artifacts, detects meaningful visual changes, and maps those changes to physical components.

---

## Current Version

**v0.2.0 - Automatic Component Detection**

### Current Capabilities

- Image loading and validation
- Image preprocessing
- Illumination normalization
- ORB feature extraction
- Feature matching
- RANSAC homography estimation
- Image registration
- Registration validity masking
- Pixel-level difference analysis
- Structural Similarity (SSIM)
- Registration residual suppression
- False-positive border filtering
- Changed-region localization
- Component registry
- Bounding-box based component definitions
- Region-to-component mapping
- Per-component CHANGED / UNCHANGED verification
- Component overlap measurement
- Deterministic verification confidence scoring
- Validation artifact generation

---

## Current Pipeline

    Reference Image
          |
          v
    Image Validation
          |
          v
    Preprocessing
          |
          v
    Image Registration
          |
          +-- ORB Features
          +-- Feature Matching
          +-- RANSAC Homography
          +-- Valid Registration Mask
          |
          v
    Aligned Inspection Image
          |
          v
    Change Detection
          |
          +-- Pixel Difference
          +-- SSIM
          +-- Residual Suppression
          +-- Morphological Filtering
          |
          v
    Changed Regions
          |
          v
    Component Registry
          |
          v
    Region-to-Component Mapping
          |
          v
    Component Verification
          |
          +-- CHANGED
          +-- UNCHANGED
          +-- Overlap Ratio
          +-- Confidence Score

---

## Validated Results

### Image Registration

Registration validation achieved:

    Matches total: 1410
    Matches used:  649
    Inliers:       556
    Inlier ratio:  0.857

    Error before:  12.780
    Error after:   2.177
    Improvement:   82.97%

Result:

    REDGUARD v0.0.3 REGISTRATION: PASS

### Change Detection

End-to-end change detection achieved:

    Registration inlier ratio: 0.823
    Global similarity:          0.9573
    Changed regions:            1
    Changed pixels:             1124
    Changed area ratio:         0.2277%

Localized region:

    x=137
    y=136
    width=47
    height=24

Result:

    REDGUARD v0.0.4 CHANGE DETECTION: PASS

### Component-Level Verification

The localized change was successfully mapped to the correct physical component:

    Q14  transistor          CHANGED
    R27  resistor            UNCHANGED
    C08  capacitor           UNCHANGED
    U03  integrated_circuit  UNCHANGED

Q14 verification:

    overlap:    53.71%
    confidence: 0.756

Result:

    REDGUARD v0.1.0 COMPONENT VERIFICATION: PASS

The current confidence score is deterministic and based on spatial overlap. It is not an ML probability.

---

## Test Status

    58 passed

The test suite covers the current foundation, preprocessing, registration, change detection, component models, and component verification layers.

---

## Version History

| Version | Milestone | Status |
| --- | --- | --- |
| v0.0.1 | Project Foundation | Complete |
| v0.0.2 | Image Preprocessing | Complete |
| v0.0.3 | Image Registration | Complete |
| v0.0.4 | Baseline Change Detection | Complete |
| v0.1.0 | Component-Level Verification | Complete |
| v0.2.0 | Automatic Component Detection | Current |
| v0.3.0 | Component Visual Fingerprinting | Planned |
| v0.4.0 | Replacement / Identity Verification | Planned |
| v0.5.0 | Anomaly Classification | Planned |
| v0.6.0 | Inspection API & Persistence | Planned |
| v0.7.0 | Monitoring & Observability | Planned |
| v0.8.0 | Industrial Evaluation Pipeline | Planned |
| v0.9.0 | Production Hardening | Planned |
| v0.10.0 | Release Candidate | Planned |
| v1.0.0 | Production Release | Planned |

---

## Next Milestone - v0.2.0

**Automatic Component Detection**

The current component registry uses predefined component locations.

The next milestone will focus on automatically detecting components from images.

Planned work:

- Component detection architecture
- Dataset and annotation pipeline
- Object detection model integration
- Automatic bounding-box generation
- Component type classification
- Detection confidence scoring
- Precision and recall evaluation
- mAP evaluation
- Integration with the existing verification pipeline

Target flow:

    Image
      |
      v
    Automatic Component Detection
      |
      +-- Transistor
      +-- Resistor
      +-- Capacitor
      +-- Integrated Circuit
      |
      v
    Detected Component Registry
      |
      v
    Change Detection
      |
      v
    Component-Level Verification

---

## Long-Term Goal

RedGuard AI aims to evolve from visual difference detection into a physical-state verification platform capable of answering:

- Did the physical system change?
- Which component changed?
- Was a component removed?
- Was a component moved?
- Was a component replaced?
- Is the replacement visually consistent with the expected component?
- Does the physical system still match its trusted reference state?

Potential application areas include industrial inspection, electronics, infrastructure, maintenance, and security-sensitive physical systems.

---

## License

MIT License
