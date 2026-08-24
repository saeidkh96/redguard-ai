import asyncio

import httpx

from redguard.api.app import create_app
from redguard.persistence import InspectionRepository
from redguard.services import ArtifactService, InspectionService


def test_readiness_endpoint_reports_orchestration(tmp_path):
    service = InspectionService(
        InspectionRepository(tmp_path / "history.json"),
        ArtifactService(tmp_path / "artifacts"),
    )
    app = create_app(service)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert "end-to-end-orchestration" in body["capabilities"]

    asyncio.run(scenario())
