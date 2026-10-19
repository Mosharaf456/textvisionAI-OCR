"""Use Case 1: Document Scanner - Digitize printed documents into editable text and Searchable PDF.

Applies deskewing, shadow removal, Sauvola binarization, Recursive XY-Cut layout analysis,
and produces plain text, structured Markdown, and Searchable PDF.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas

from source.core.factory import get_ocr_engine
from source.core.logger import create_request_id, get_logger
from source.core.types import OCRResult, TextLine
from source.preprocessing.binarization import remove_shadows, sauvola_threshold
from source.preprocessing.deskew import Deskewer
from source.preprocessing.layout import RecursiveXYCut

logger = get_logger("use_cases.document_scanner")


class DocumentScanner:
    """Digitizes scanned paper documents into searchable and editable formats."""

    def __init__(
        self,
        auto_deskew: bool = True,
        auto_remove_shadows: bool = True,
        apply_binarization: bool = False,
    ):
        self.auto_deskew = auto_deskew
        self.auto_remove_shadows = auto_remove_shadows
        self.apply_binarization = apply_binarization

        self.deskewer = Deskewer()
        self.xy_cut = RecursiveXYCut()
        self.engine = get_ocr_engine()

    def process(
        self,
        image_input: Union[str, Path, np.ndarray],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process document image through preprocessing, OCR, and reading order reconstruction."""
        req_id = request_id or create_request_id()

        # Load image
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            img = cv2.imread(path_str)
            if img is None:
                raise ValueError(f"Could not read image file: {path_str}")
        else:
            img = image_input.copy()

        h, w = img.shape[:2]
        processed_img = img
        applied_angle = 0.0

        # Step 1: Deskew
        if self.auto_deskew:
            processed_img, applied_angle = self.deskewer(processed_img)

        # Step 2: Shadow removal
        if self.auto_remove_shadows:
            processed_img = remove_shadows(processed_img)

        # Step 3: Binarization (optional for heavily degraded papers)
        if self.apply_binarization:
            bin_img = sauvola_threshold(processed_img)
            # Convert back to 3-channel for OCR engine
            processed_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

        # Step 4: Core OCR
        raw_result = self.engine.predict(processed_img, request_id=req_id)

        # Step 5: Multi-column reading order reconstruction via Recursive XY-Cut
        sorted_lines = self.xy_cut.sort_reading_order(raw_result.lines)
        ordered_result = OCRResult(
            lines=sorted_lines,
            image_shape=(h, w),
            latency_ms=raw_result.latency_ms,
            preprocessing_latency_ms=raw_result.preprocessing_latency_ms,
            inference_latency_ms=raw_result.inference_latency_ms,
            metadata={
                **raw_result.metadata,
                "deskew_angle": applied_angle,
                "reading_order_reconstructed": True,
            },
        )

        # Step 6: Generate text representations
        plain_text = ordered_result.full_text
        markdown_text = self._generate_markdown(sorted_lines)

        return {
            "request_id": req_id,
            "plain_text": plain_text,
            "markdown": markdown_text,
            "ocr_result": ordered_result,
            "preprocessed_image": processed_img,
            "skew_angle": applied_angle,
            "total_lines": len(sorted_lines),
            "confidence": round(ordered_result.mean_confidence, 4),
        }

    def _generate_markdown(self, lines: List[TextLine]) -> str:
        """Format recognized text as Markdown with font-size-based heading inference."""
        if not lines:
            return ""

        # Calculate line height distribution to distinguish headings from body text
        heights = [l.bbox.height for l in lines]
        median_h = float(np.median(heights)) if heights else 20.0
        min_h = float(np.min(heights)) if heights else 20.0

        md_output = []
        current_block = -1

        for line in lines:
            # Paragraph separation when block changes
            if line.block_id != current_block and current_block != -1:
                md_output.append("")
            current_block = line.block_id

            h = line.bbox.height
            text = line.text.strip()

            is_all_caps = text.isupper() and len(text) >= 4 and not text.isdigit()
            if h >= median_h * 1.25 or (is_all_caps and len(text.split()) <= 6):
                # Top-level Title / Heading 1
                md_output.append(f"# {text}")
            elif h >= median_h * 1.12:
                # Subheading / Heading 2
                md_output.append(f"## {text}")
            else:
                md_output.append(text)

        return "\n".join(md_output)

    def export_searchable_pdf(
        self,
        image_input: Union[str, Path, np.ndarray],
        output_pdf_path: Union[str, Path],
        ocr_result: Optional[OCRResult] = None,
    ) -> Path:
        """Create a Searchable PDF with original image background and invisible text layer."""
        output_path = Path(output_pdf_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run process if OCR result not pre-supplied
        if ocr_result is None:
            processed = self.process(image_input)
            ocr_result = processed["ocr_result"]
            img_bgr = processed["preprocessed_image"]
        else:
            if isinstance(image_input, (str, Path)):
                img_bgr = cv2.imread(str(image_input))
            else:
                img_bgr = image_input

        img_h, img_w = img_bgr.shape[:2]

        # Convert image to JPEG bytes for ReportLab
        _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_bytes = io.BytesIO(buffer)

        # PDF points equal pixels at 72 dpi; we set canvas size to match image pixels exactly
        c = canvas.Canvas(str(output_path), pagesize=(img_w, img_h))

        # 1. Draw image background
        from reportlab.lib.utils import ImageReader
        img_reader = ImageReader(img_bytes)
        c.drawImage(img_reader, 0, 0, width=img_w, height=img_h)

        # 2. Draw invisible text overlay precisely matching OCR bounding boxes
        # ReportLab (0,0) is at bottom-left, while image (0,0) is at top-left
        # Make text invisible: transparent text rendering mode
        c.setFillColor(Color(0, 0, 0, alpha=0.0))

        for line in ocr_result.lines:
            x_min = line.bbox.x_min
            # Invert Y coordinate for PDF coordinate space
            y_pdf = img_h - line.bbox.y_max
            line_h = line.bbox.height
            line_w = line.bbox.width

            # Approximate font size to fit height
            font_size = max(6.0, line_h * 0.8)
            c.setFont("Helvetica", font_size)

            # Draw text
            c.drawString(x_min, y_pdf, line.text)

        c.showPage()
        c.save()

        logger.info("Searchable PDF created", path=str(output_path), lines=len(ocr_result.lines))
        return output_path
