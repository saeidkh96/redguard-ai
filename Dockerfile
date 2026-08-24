FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REDGUARD_ARTIFACTS_DIR=/app/artifacts \
    REDGUARD_DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY models ./models

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN mkdir -p /app/artifacts /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import json, urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3); d=json.load(r); raise SystemExit(0 if r.status == 200 and d.get('status') == 'ready' else 1)"

CMD ["python", "-m", "uvicorn", "redguard.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
