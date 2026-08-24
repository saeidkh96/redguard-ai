import asyncio

import httpx

from redguard.api import create_app
from redguard.persistence import InspectionRepository
from redguard.services import ArtifactService, InspectionService


def build_app(tmp_path):
    service = InspectionService(
        InspectionRepository(tmp_path / "history.json"),
        ArtifactService(tmp_path / "artifacts"),
    )
    return create_app(service)


def payload():
    return {
        "component_id": "Q14",
        "component_type": "transistor",
        "change_score": 0.9,
        "fingerprint_similarity": 0.3,
        "anomaly_score": 0.95,
        "edge_difference": 0.75,
        "texture_difference": 0.7,
        "reference_consensus": 0.0,
    }


async def request(app, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def test_health(tmp_path):
    response = asyncio.run(request(build_app(tmp_path), "GET", "/health"))
    assert response.status_code == 200


def test_create_get_and_list(tmp_path):
    app = build_app(tmp_path)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            created = await client.post("/api/v1/inspections", json=payload())
            assert created.status_code == 201
            body = created.json()
            assert body["decision"] == "FAIL"

            fetched = await client.get(
                f'/api/v1/inspections/{body["inspection_id"]}'
            )
            assert fetched.status_code == 200

            history = await client.get("/api/v1/inspections")
            assert history.status_code == 200
            assert len(history.json()) == 1

    asyncio.run(scenario())


def test_missing_is_404(tmp_path):
    response = asyncio.run(
        request(build_app(tmp_path), "GET", "/api/v1/inspections/missing")
    )
    assert response.status_code == 404
