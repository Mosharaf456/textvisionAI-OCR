"""Integration tests for all 4 industrial OCR use cases."""

from __future__ import annotations

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest
from reportlab.pdfgen import canvas

from source.schemas.alpr_schema import PlateBoundingBox, PlateDetection
from source.use_cases.document_scanner import DocumentScanner
from source.use_cases.pdf_image_reader import PDFImageReader
from source.use_cases.structured_extractor import StructuredExtractor
from source.use_cases.vehicle_alpr import TemporalPlateTracker, VehicleALPR


def test_use_case_1_document_scanner():
    scanner = DocumentScanner(auto_deskew=False, auto_remove_shadows=False)

    # Synthetic document image
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.putText(img, "OFFICIAL NOTICE", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "This document confirms OCR scanning.", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    result = scanner.process(img)
    assert result["total_lines"] >= 2
    assert "# " in result["markdown"]
    assert "NOTICE" in result["markdown"]

    # Test Searchable PDF export
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = Path(tmp.name)

    exported = scanner.export_searchable_pdf(img, pdf_path, result["ocr_result"])
    assert exported.exists()
    assert exported.stat().st_size > 1000
    pdf_path.unlink()


def test_use_case_2_hybrid_pdf_reader():
    reader = PDFImageReader()

    # Generate a pure digital PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = Path(tmp.name)

    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "This is digital vector text that should be extracted in milliseconds.")
    c.drawString(100, 700, "Second line of digital stream content for hybrid testing.")
    c.save()

    res = reader.read(pdf_path, force_ocr=False)
    assert res["file_type"] == "pdf"
    assert res["total_pages"] == 1
    # Should use digital fastpath without running OCR!
    assert res["pages"][0]["extraction_method"] == "digital_fastpath"
    assert "digital vector text" in res["full_text"]
    pdf_path.unlink()


def test_use_case_3_structured_invoice_extraction():
    extractor = StructuredExtractor()

    # Synthetic invoice image
    img = np.full((500, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, "INVOICE # INV-9001", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Date: 2026-10-19", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Bill To: Acme Corp", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Description Qty Price Amount", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Widget A 2 50.00 100.00", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Subtotal: 100.00", (350, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Tax: 10.00", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Total: 110.00", (350, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    inv = extractor.extract_invoice(img)
    assert inv.invoice_number == "INV-9001"
    assert inv.invoice_date == "2026-10-19"
    assert inv.totals.subtotal == 100.0
    assert inv.totals.total == 110.0
    assert inv.is_math_valid is True


def test_use_case_4_alpr_and_temporal_tracking():
    tracker = TemporalPlateTracker()

    # Simulate 5 consecutive CCTV frames of the same vehicle
    # 4 frames return "DHAKA METRO GA 11-2233", 1 frame has motion blur noise "DHAKA METRO GA 11-2288"
    good_det = PlateDetection(
        plate_text="DHAKA METRO GA 11-2233",
        normalized_text="DHAKA METRO GA 11-2233",
        confidence=0.95,
        format_name="BD_DIGITAL_METRO",
        is_format_valid=True,
    )
    noisy_det = PlateDetection(
        plate_text="DHAKA METRO GA 11-2288",
        normalized_text="DHAKA METRO GA 11-2288",
        confidence=0.60,
        format_name="BD_DIGITAL_METRO",
        is_format_valid=True,
    )

    track_id = 42
    tracker.update(track_id, good_det)
    tracker.update(track_id, good_det)
    tracker.update(track_id, noisy_det)  # transient error
    tracker.update(track_id, good_det)
    final_event = tracker.update(track_id, good_det)

    assert final_event is not None
    assert final_event.track_id == 42
    # Temporal voting should eliminate the noisy frame and pick the true consensus!
    assert final_event.plate_text == "DHAKA METRO GA 11-2233"
    assert final_event.total_frames_observed == 5
    assert final_event.is_valid_format is True


def test_use_case_4_alpr_process_frame_synthetic():
    from source.use_cases.vehicle_alpr import VehicleALPR
    alpr = VehicleALPR()
    # Create synthetic frame with plate text
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 180
    cv2.rectangle(frame, (200, 300), (440, 360), (255, 255, 255), -1)
    cv2.rectangle(frame, (200, 300), (440, 360), (0, 0, 0), 2)
    cv2.putText(frame, "DHAKA METRO GA 11-2233", (210, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    detections = alpr.process_frame(frame)
    assert len(detections) >= 1
    assert any("DHAKA" in d.plate_text for d in detections)

