"""FastAPI REST API Service for CPU-based Production OCR Pipeline."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from source.core.factory import get_ocr_engine
from source.core.logger import create_request_id, get_logger, setup_logger
from source.schemas.alpr_schema import PlateDetection
from source.schemas.id_card_schema import IDCardExtraction
from source.schemas.invoice_schema import InvoiceExtraction
from source.use_cases.document_scanner import DocumentScanner
from source.use_cases.pdf_image_reader import PDFImageReader
from source.use_cases.structured_extractor import StructuredExtractor
from source.use_cases.vehicle_alpr import VehicleALPR

setup_logger()
logger = get_logger("api")

app = FastAPI(
    title="TextVision AI",
    description="TextVision AI — Intelligent text recognition and extraction.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global metrics tracker
_METRICS = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "total_latency_ms": 0.0,
}

# Lazily initialized use cases
_scanner: Optional[DocumentScanner] = None
_pdf_reader: Optional[PDFImageReader] = None
_extractor: Optional[StructuredExtractor] = None
_alpr: Optional[VehicleALPR] = None


def get_scanner() -> DocumentScanner:
    global _scanner
    if _scanner is None:
        _scanner = DocumentScanner()
    return _scanner


def get_pdf_reader() -> PDFImageReader:
    global _pdf_reader
    if _pdf_reader is None:
        _pdf_reader = PDFImageReader()
    return _pdf_reader


def get_extractor() -> StructuredExtractor:
    global _extractor
    if _extractor is None:
        _extractor = StructuredExtractor()
    return _extractor


def get_alpr() -> VehicleALPR:
    global _alpr
    if _alpr is None:
        _alpr = VehicleALPR()
    return _alpr


async def _decode_upload_image(file: UploadFile) -> np.ndarray:
    """Read uploaded bytes and decode into OpenCV BGR numpy array."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file or format could not be decoded")
    return img


@app.get("/", tags=["UI"], include_in_schema=False)
async def serve_ui() -> Any:
    """Serve the Web UI for interactive OCR testing."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "CPU OCR API is running. UI not found in static folder."}


@app.get("/api/v1/sample/{sample_name}", tags=["Samples"])
async def get_sample_asset(sample_name: str) -> FileResponse:
    """Serve sample assets for quick one-click UI testing."""
    base_res = Path(__file__).resolve().parent.parent.parent / "resources"
    mapping = {
        "document": base_res / "sample_documents" / "sample_document.png",
        "invoice": base_res / "sample_documents" / "sample_invoice.png",
        "id_card": base_res / "sample_documents" / "sample_id_card.png",
        "cctv": base_res / "sample_cctv" / "sample_cctv_frame.png",
    }
    sample_path = mapping.get(sample_name.lower())
    if not sample_path or not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{sample_name}' not found")
    return FileResponse(sample_path, media_type="image/png")


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Returns engine health, ONNX Runtime status, and thread configuration."""
    engine = get_ocr_engine()
    return {
        "status": "healthy",
        "engine": engine.name,
        "backend": "onnxruntime-cpu",
        "timestamp": time.time(),
    }


@app.get("/metrics", tags=["System"])
async def get_metrics() -> Dict[str, Any]:
    """Returns runtime latency and throughput metrics."""
    reqs = _METRICS["requests_total"]
    avg_latency = (_METRICS["total_latency_ms"] / reqs) if reqs > 0 else 0.0
    return {
        "requests_total": _METRICS["requests_total"],
        "requests_success": _METRICS["requests_success"],
        "requests_failed": _METRICS["requests_failed"],
        "avg_latency_ms": round(avg_latency, 2),
    }


@app.post("/api/v1/scan", tags=["Use Case 1: Document Scanner"])
async def scan_document(
    file: UploadFile = File(..., description="Document image (PNG, JPG, TIFF)"),
    deskew: bool = Query(True, description="Enable automatic deskewing"),
    export_pdf: bool = Query(False, description="Return searchable PDF file directly"),
) -> Any:
    """Use Case 1: Digitize printed document with reading order preservation."""
    req_id = create_request_id()
    t0 = time.perf_counter()
    _METRICS["requests_total"] += 1

    try:
        img = await _decode_upload_image(file)
        scanner = get_scanner()
        res = scanner.process(img, request_id=req_id)

        if export_pdf:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            scanner.export_searchable_pdf(img, tmp_path, res["ocr_result"])
            with open(tmp_path, "rb") as f:
                pdf_bytes = f.read()
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=searchable_{file.filename}.pdf"},
            )

        duration = (time.perf_counter() - t0) * 1000.0
        _METRICS["requests_success"] += 1
        _METRICS["total_latency_ms"] += duration

        return {
            "request_id": req_id,
            "filename": file.filename,
            "plain_text": res["plain_text"],
            "markdown": res["markdown"],
            "total_lines": res["total_lines"],
            "confidence": res["confidence"],
            "skew_angle": res["skew_angle"],
            "duration_ms": round(duration, 2),
            "lines": [l.to_dict() for l in res["ocr_result"].lines],
        }
    except Exception as e:
        _METRICS["requests_failed"] += 1
        logger.error("Scan failed", error=str(e), request_id=req_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/extract-invoice", response_model=InvoiceExtraction, tags=["Use Case 3: Invoices"])
async def extract_invoice(
    file: UploadFile = File(..., description="Invoice image (PNG, JPG, TIFF)"),
) -> InvoiceExtraction:
    """Use Case 3: Extract structured invoice metadata, line items, and validate arithmetic."""
    req_id = create_request_id()
    try:
        img = await _decode_upload_image(file)
        extractor = get_extractor()
        return extractor.extract_invoice(img, request_id=req_id)
    except Exception as e:
        logger.error("Invoice extraction failed", error=str(e), request_id=req_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/extract-id", response_model=IDCardExtraction, tags=["Use Case 3: ID Cards"])
async def extract_id_card(
    file: UploadFile = File(..., description="ID card or passport image"),
) -> IDCardExtraction:
    """Use Case 3: Extract structured identity fields and MRZ zones from IDs and Passports."""
    req_id = create_request_id()
    try:
        img = await _decode_upload_image(file)
        extractor = get_extractor()
        return extractor.extract_id_card(img, request_id=req_id)
    except Exception as e:
        logger.error("ID extraction failed", error=str(e), request_id=req_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/alpr", response_model=List[PlateDetection], tags=["Use Case 4: CCTV ALPR"])
async def recognize_license_plates(
    file: UploadFile = File(..., description="CCTV video frame image"),
) -> List[PlateDetection]:
    """Use Case 4: Detect and recognize vehicle license plates in CCTV frame."""
    req_id = create_request_id()
    try:
        img = await _decode_upload_image(file)
        alpr_engine = get_alpr()
        return alpr_engine.process_frame(img, request_id=req_id)
    except Exception as e:
        logger.error("ALPR failed", error=str(e), request_id=req_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/read", tags=["Use Case 2: Multi-Format & PDF Ingestion"])
async def read_document_or_pdf(
    file: UploadFile = File(..., description="PDF or image file (PDF, PNG, JPG, TIFF, WebP)"),
    force_ocr: bool = Query(False, description="Force raster OCR on all pages"),
) -> Dict[str, Any]:
    """Use Case 2: Ingest PDF (digital text fastpath + raster fallback) or image file."""
    import tempfile
    req_id = create_request_id()
    suffix = Path(file.filename or "doc.tmp").suffix
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        reader = get_pdf_reader()
        result = reader.read(tmp_path, force_ocr=force_ocr, request_id=req_id)
        tmp_path.unlink(missing_ok=True)
        return result
    except Exception as e:
        logger.error("Read failed", error=str(e), request_id=req_id)
        raise HTTPException(status_code=500, detail=str(e))

