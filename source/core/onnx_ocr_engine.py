"""ONNX Runtime OCR Engine implementation using PP-OCRv4 architecture.

Optimized for multi-core Intel/AMD CPUs with AVX2 vectorization.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from source.core.base_engine import BaseOCREngine
from source.core.logger import create_request_id, get_logger
from source.core.types import BoundingBox, OCRResult, TextLine

logger = get_logger("onnx_ocr_engine")


class ONNXOCREngine(BaseOCREngine):
    """High-throughput CPU OCR engine using ONNX Runtime.

    Leverages DBNet detection, direction angle classification, and SVTR recognition.
    """

    def __init__(
        self,
        det_model_path: Optional[Union[str, Path]] = None,
        rec_model_path: Optional[Union[str, Path]] = None,
        cls_model_path: Optional[Union[str, Path]] = None,
        intra_op_num_threads: int = 4,
        inter_op_num_threads: int = 2,
        use_angle_cls: bool = True,
        unclip_ratio: float = 1.6,
        box_thresh: float = 0.5,
        text_score: float = 0.5,
    ):
        self.intra_op_num_threads = intra_op_num_threads
        self.inter_op_num_threads = inter_op_num_threads
        self.use_angle_cls = use_angle_cls
        self.unclip_ratio = unclip_ratio
        self.box_thresh = box_thresh
        self.text_score = text_score

        # Prepare kwargs for RapidOCR
        rapid_kwargs: Dict[str, Any] = {
            "use_angle_cls": use_angle_cls,
            "unclip_ratio": unclip_ratio,
            "box_thresh": box_thresh,
            "text_score": text_score,
        }

        if det_model_path and Path(det_model_path).exists():
            rapid_kwargs["det_model_path"] = str(det_model_path)
        if rec_model_path and Path(rec_model_path).exists():
            rapid_kwargs["rec_model_path"] = str(rec_model_path)
        if cls_model_path and Path(cls_model_path).exists():
            rapid_kwargs["cls_model_path"] = str(cls_model_path)

        logger.info(
            "Initializing ONNX OCR Engine",
            intra_threads=intra_op_num_threads,
            inter_threads=inter_op_num_threads,
            use_angle_cls=use_angle_cls,
        )

        self._engine = RapidOCR(**rapid_kwargs)
        self.warmup()

    @property
    def name(self) -> str:
        return "onnx_ppocr_v4"

    def warmup(self) -> None:
        """Run warm-up inference to compile graph execution providers."""
        try:
            dummy = np.full((128, 256, 3), 255, dtype=np.uint8)
            cv2.putText(dummy, "WARMUP", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            self._engine(dummy)
            logger.debug("ONNX OCR engine warmed up successfully")
        except Exception as e:
            logger.warning("Warmup failed", error=str(e))

    def _load_image(self, image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """Standardize input to uint8 BGR numpy array."""
        if isinstance(image, (str, Path)):
            path_str = str(image)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image not found: {path_str}")
            img_bgr = cv2.imread(path_str)
            if img_bgr is None:
                raise ValueError(f"Failed to read image at: {path_str}")
            return img_bgr
        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def predict(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Run OCR detection and recognition on a single image."""
        req_id = request_id or create_request_id()
        t0 = time.perf_counter()

        img_bgr = self._load_image(image)
        h, w = img_bgr.shape[:2]

        t_prep = time.perf_counter() - t0
        t_infer_start = time.perf_counter()

        raw_results, elapse = self._engine(img_bgr, **kwargs)
        t_infer = time.perf_counter() - t_infer_start
        t_total = time.perf_counter() - t0

        lines: List[TextLine] = []
        if raw_results:
            for idx, item in enumerate(raw_results):
                # item format: [box_points, text, confidence]
                poly_pts = item[0]
                text = str(item[1]).strip()
                conf = float(item[2])

                if not text:
                    continue

                bbox = BoundingBox.from_polygon(poly_pts)
                lines.append(
                    TextLine(
                        text=text,
                        confidence=conf,
                        bbox=bbox,
                        line_id=idx,
                    )
                )

        logger.info(
            "OCR inference completed",
            request_id=req_id,
            lines_found=len(lines),
            total_ms=round(t_total * 1000, 2),
            infer_ms=round(t_infer * 1000, 2),
        )

        return OCRResult(
            lines=lines,
            image_shape=(h, w),
            latency_ms=t_total * 1000.0,
            preprocessing_latency_ms=t_prep * 1000.0,
            inference_latency_ms=t_infer * 1000.0,
            metadata={
                "request_id": req_id,
                "engine": self.name,
                "model_timings": elapse if isinstance(elapse, list) else [],
            },
        )

    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[OCRResult]:
        """Process batch of images sequentially on CPU (avoids thread contention)."""
        req_id = request_id or create_request_id()
        results: List[OCRResult] = []
        for i, img in enumerate(images):
            sub_id = f"{req_id}-page-{i+1}"
            results.append(self.predict(img, request_id=sub_id, **kwargs))
        return results
