"""Unit tests for FastAPI REST API endpoints."""

from __future__ import annotations

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from source.api.app import app

client = TestClient(app)


def _create_test_image_bytes(text: str = "API TEST") -> io.BytesIO:
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    _, buffer = cv2.imencode(".png", img)
    return io.BytesIO(buffer.tobytes())


def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["backend"] == "onnxruntime-cpu"


def test_api_metrics():
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "requests_total" in data


def test_api_scan_document():
    img_bytes = _create_test_image_bytes("SCAN TEST DOC")
    res = client.post(
        "/api/v1/scan",
        files={"file": ("test_doc.png", img_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "plain_text" in data
    assert "markdown" in data
    assert data["total_lines"] >= 1


def test_api_alpr():
    img_bytes = _create_test_image_bytes("DHAKA METRO GA 11-2233")
    res = client.post(
        "/api/v1/alpr",
        files={"file": ("plate.png", img_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)


def test_api_serve_ui():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "TextVision AI" in res.text


def test_api_sample_endpoint():
    res = client.get("/api/v1/sample/document")
    assert res.status_code == 200
    assert "image/png" in res.headers["content-type"]


def test_api_read_endpoint():
    img_bytes = _create_test_image_bytes("HYBRID PDF TEST")
    res = client.post(
        "/api/v1/read",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["file_type"] == "image"
    assert "full_text" in data

