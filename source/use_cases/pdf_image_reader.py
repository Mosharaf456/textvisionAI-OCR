"""Use Case 2: Multi-Format Image and PDF Ingestion.

Hybrid document engine:
- Direct vector text extraction on digital PDFs (< 15ms/page).
- Parallelized multi-threaded raster OCR fallback for scanned pages.
- Native ingestion of JPG, PNG, TIFF, WebP, BMP images.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import pypdf
from pdf2image import convert_from_path
from PIL import Image

from source.core.factory import get_ocr_engine
from source.core.logger import create_request_id, get_logger
from source.core.types import BoundingBox, OCRResult, TextLine

logger = get_logger("use_cases.pdf_image_reader")


class PDFImageReader:
    """High-throughput reader for digital PDFs, scanned PDFs, and multi-format image files."""

    def __init__(
        self,
        dpi: int = 200,
        digital_text_min_chars: int = 50,
        max_workers: int = 4,
    ):
        self.dpi = dpi
        self.digital_text_min_chars = digital_text_min_chars
        self.max_workers = max_workers
        self.engine = get_ocr_engine()

    def read(
        self,
        file_path: Union[str, Path],
        force_ocr: bool = False,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read text from an image or PDF file.

        Args:
            file_path: Path to target file.
            force_ocr: If True, bypass digital text extraction and force raster OCR on all pages.
            request_id: Optional tracing ID.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        req_id = request_id or create_request_id()
        suffix = path.suffix.lower()

        logger.info("Reading document", path=str(path), format=suffix, request_id=req_id)

        if suffix == ".pdf":
            return self._read_pdf(path, force_ocr=force_ocr, request_id=req_id)
        elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"]:
            return self._read_image(path, request_id=req_id)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _read_image(self, path: Path, request_id: str) -> Dict[str, Any]:
        """Process a single image file."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to decode image: {path}")

        ocr_res = self.engine.predict(img, request_id=request_id)
        return {
            "request_id": request_id,
            "file_type": "image",
            "file_name": path.name,
            "total_pages": 1,
            "full_text": ocr_res.full_text,
            "pages": [
                {
                    "page_number": 1,
                    "extraction_method": "ocr",
                    "text": ocr_res.full_text,
                    "confidence": ocr_res.mean_confidence,
                    "lines_count": len(ocr_res.lines),
                    "ocr_result": ocr_res,
                }
            ],
            "total_latency_ms": round(ocr_res.latency_ms, 2),
        }

    def _read_pdf(self, path: Path, force_ocr: bool, request_id: str) -> Dict[str, Any]:
        """Hybrid processing of PDF file (digital stream vs raster OCR fallback)."""
        reader = pypdf.PdfReader(str(path))
        total_pages = len(reader.pages)
        pages_output: List[Dict[str, Any]] = [{} for _ in range(total_pages)]

        # Determine which pages require raster OCR
        pages_to_ocr: List[int] = []

        if not force_ocr:
            for idx, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                if len(text.strip()) >= self.digital_text_min_chars:
                    pages_output[idx] = {
                        "page_number": idx + 1,
                        "extraction_method": "digital_fastpath",
                        "text": text.strip(),
                        "confidence": 1.0,
                        "lines_count": len(text.splitlines()),
                        "ocr_result": None,
                    }
                else:
                    pages_to_ocr.append(idx)
        else:
            pages_to_ocr = list(range(total_pages))

        # If any pages need OCR, render them
        if pages_to_ocr:
            logger.info(
                "Raster OCR required for scanned pages",
                scanned_pages=[p + 1 for p in pages_to_ocr],
                total_pages=total_pages,
            )

            # Convert only required pages to images to save RAM and CPU
            # pdf2image first_page / last_page are 1-indexed
            def process_scanned_page(page_idx: int) -> Dict[str, Any]:
                images = convert_from_path(
                    str(path),
                    dpi=self.dpi,
                    first_page=page_idx + 1,
                    last_page=page_idx + 1,
                )
                if not images:
                    return {
                        "page_number": page_idx + 1,
                        "extraction_method": "failed",
                        "text": "",
                        "confidence": 0.0,
                        "lines_count": 0,
                    }

                pil_img = images[0]
                # Convert PIL to BGR numpy
                bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                page_req_id = f"{request_id}-p{page_idx+1}"
                ocr_res = self.engine.predict(bgr, request_id=page_req_id)

                return {
                    "page_number": page_idx + 1,
                    "extraction_method": "raster_ocr",
                    "text": ocr_res.full_text,
                    "confidence": ocr_res.mean_confidence,
                    "lines_count": len(ocr_res.lines),
                    "ocr_result": ocr_res,
                }

            # Process raster pages in parallel worker pool
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(pages_to_ocr))) as executor:
                futures = {executor.submit(process_scanned_page, p): p for p in pages_to_ocr}
                for f in futures:
                    page_idx = futures[f]
                    try:
                        pages_output[page_idx] = f.result()
                    except Exception as e:
                        logger.error("Failed to OCR page", page=page_idx + 1, error=str(e))
                        pages_output[page_idx] = {
                            "page_number": page_idx + 1,
                            "extraction_method": "error",
                            "error": str(e),
                            "text": "",
                            "confidence": 0.0,
                        }

        # Combine text across all pages
        full_text_parts = [p.get("text", "") for p in pages_output if p.get("text")]
        full_document_text = "\n\n--- Page Break ---\n\n".join(full_text_parts)

        return {
            "request_id": request_id,
            "file_type": "pdf",
            "file_name": path.name,
            "total_pages": total_pages,
            "full_text": full_document_text,
            "pages": pages_output,
        }
