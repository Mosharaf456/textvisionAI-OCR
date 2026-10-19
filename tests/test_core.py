"""Unit tests for core OCR types, model manager, and inference engine."""

from __future__ import annotations

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from source.core.factory import get_ocr_engine
from source.core.model_manager import ModelManager
from source.core.types import BoundingBox, OCRResult, Point, TextLine


def test_bounding_box_operations():
    # Construct from 4 points
    pts = [[10, 20], [110, 20], [110, 60], [10, 60]]
    bbox = BoundingBox.from_polygon(pts)

    assert bbox.x_min == 10.0
    assert bbox.x_max == 110.0
    assert bbox.y_min == 20.0
    assert bbox.y_max == 60.0
    assert bbox.width == 100.0
    assert bbox.height == 40.0
    assert bbox.area == 4000.0
    assert bbox.center == (60.0, 40.0)
    assert bbox.to_xywh() == (10, 20, 100, 40)


def test_text_line_and_ocr_result():
    bbox = BoundingBox.from_xywh(0, 0, 50, 20)
    line = TextLine(text="TEST LINE", confidence=0.98, bbox=bbox, line_id=0)

    result = OCRResult(lines=[line], image_shape=(100, 100), latency_ms=15.5)
    assert result.full_text == "TEST LINE"
    assert result.mean_confidence == 0.98

    d = result.to_dict()
    assert d["total_lines"] == 1
    assert "breakdown_ms" in d


def test_model_manager_sha256():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"Antigravity CPU OCR Test Content")
        tmp_path = Path(tmp.name)

    digest = ModelManager.calculate_sha256(tmp_path)
    assert len(digest) == 64
    tmp_path.unlink()


def test_engine_inference_on_synthetic_image():
    engine = get_ocr_engine()
    assert engine.name == "onnx_ppocr_v4"

    # Create synthetic image
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST OCR", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    res = engine.predict(img)
    assert len(res.lines) >= 1
    assert "TEST" in res.full_text
    assert res.mean_confidence > 0.8
