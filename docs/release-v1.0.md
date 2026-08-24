# RedGuard AI v1.0.0 Release Validation

RedGuard AI v1.0.0 is the first portfolio-grade production release of the project.
It consolidates the validated inspection pipeline developed from v0.0.1 through v0.10.0.

## Release scope

The release includes:

- image validation, preprocessing, registration, and visual change localization;
- automatic component detection and component registry generation;
- visual fingerprinting and patch-memory anomaly detection;
- fine-grained multi-signal inspection;
- multi-reference verification;
- explainable inspection intelligence;
- a safety boundary that prevents the reasoning layer from changing deterministic risk, severity, or inspection decisions;
- service orchestration, persistence, structured artifacts, API health/readiness, Docker packaging, and CI validation.

## Validation boundary

RedGuard remains an engineering and research project. Detection benchmarks currently use the controlled synthetic RedGuard dataset, and fingerprint/anomaly thresholds are experimental. v1.0.0 does not claim certified industrial inspection performance or generalization to arbitrary hardware imagery.

## Release gates

A v1.0.0 release is considered valid when all of the following pass:

1. the complete pytest regression suite;
2. the repeatable end-to-end production pipeline validation;
3. package wheel construction;
4. API `/health` and `/ready` checks;
5. repository hygiene checks;
6. Dockerfile and Compose configuration checks;
7. CI configuration is present and committed.

## Runtime

Run the API locally:

```bash
python -m uvicorn redguard.api.app:app --host 0.0.0.0 --port 8000
```

Run with Docker Compose:

```bash
docker compose up --build
```

Health endpoints:

- `GET /health`
- `GET /ready`
