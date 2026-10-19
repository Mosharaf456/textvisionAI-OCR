"""Factory and configuration manager for OCR engines."""

from __future__ import annotations

import os
import threading
from typing import Optional
from source.core.base_engine import BaseOCREngine
from source.core.onnx_ocr_engine import ONNXOCREngine
from source.core.logger import get_logger

logger = get_logger("engine_factory")

_engine_instance: Optional[BaseOCREngine] = None
_lock = threading.Lock()


def get_ocr_engine(
    intra_op_num_threads: Optional[int] = None,
    inter_op_num_threads: Optional[int] = None,
    use_angle_cls: bool = True,
    force_new: bool = False,
) -> BaseOCREngine:
    """Get or create singleton instance of the CPU ONNX OCR Engine.

    Default thread allocation:
      intra_op_num_threads: min(4, CPU physical cores)
      inter_op_num_threads: 2
    """
    global _engine_instance

    if _engine_instance is not None and not force_new:
        return _engine_instance

    with _lock:
        if _engine_instance is not None and not force_new:
            return _engine_instance

        # Read environment variables or defaults
        intra = intra_op_num_threads or int(os.getenv("OCR_INTRA_THREADS", "4"))
        inter = inter_op_num_threads or int(os.getenv("OCR_INTER_THREADS", "2"))

        logger.info("Instantiating singleton ONNXOCREngine", intra=intra, inter=inter)
        _engine_instance = ONNXOCREngine(
            intra_op_num_threads=intra,
            inter_op_num_threads=inter,
            use_angle_cls=use_angle_cls,
        )
        return _engine_instance
