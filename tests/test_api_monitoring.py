from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from app.main import app


def test_api_exposes_prometheus_metrics_and_logs_request(caplog):
    request_id = "monitoring-test-123"
    caplog.set_level(logging.INFO, logger="app.main")
    client = TestClient(app)

    response = client.get("/health", headers={"X-Request-ID": request_id})
    metrics_response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id
    assert metrics_response.status_code == 200
    assert "http_requests_total" in metrics_response.text
    request_logs = [
        json.loads(record.message)
        for record in caplog.records
        if '"event": "api_request"' in record.message
    ]
    health_log = next(entry for entry in request_logs if entry["request_id"] == request_id)
    assert health_log["method"] == "GET"
    assert health_log["path"] == "/health"
    assert health_log["status_code"] == 200
    assert health_log["duration_ms"] >= 0
