from __future__ import annotations

from fastapi import FastAPI, HTTPException

from redguard.core.config import settings
from redguard.services import InspectionService
from .schemas import InspectionRequest, InspectionResponse


def create_app(service: InspectionService | None = None) -> FastAPI:
    app = FastAPI(title="RedGuard AI", version=settings.version)
    inspection_service = service or InspectionService()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": settings.version}

    @app.get("/ready")
    def ready() -> dict[str, object]:
        return {
            "status": "ready",
            "version": settings.version,
            "capabilities": [
                "inspection-service",
                "persistence",
                "artifact-storage",
                "safe-reasoning",
                "end-to-end-orchestration",
            ],
        }

    @app.post("/api/v1/inspections", response_model=InspectionResponse, status_code=201)
    def create_inspection(request: InspectionRequest) -> dict:
        return inspection_service.inspect(**request.model_dump()).to_dict()

    @app.get("/api/v1/inspections", response_model=list[InspectionResponse])
    def list_inspections() -> list[dict]:
        return [item.to_dict() for item in inspection_service.history()]

    @app.get("/api/v1/inspections/{inspection_id}", response_model=InspectionResponse)
    def get_inspection(inspection_id: str) -> dict:
        record = inspection_service.get(inspection_id)
        if record is None:
            raise HTTPException(status_code=404, detail="inspection not found")
        return record.to_dict()

    return app


app = create_app()
