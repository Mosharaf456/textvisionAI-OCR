"""Abstract base class interface for OCR inference engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from source.core.types import OCRResult


class BaseOCREngine(ABC):
    """Abstract Base Class for OCR engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the OCR engine backend."""
        pass

    @abstractmethod
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Run OCR detection and recognition on a single image.

        Args:
            image: Input image as numpy array (BGR or RGB), PIL Image, or file path.
            request_id: Optional trace request ID.
            **kwargs: Backend-specific tuning options.

        Returns:
            OCRResult containing text lines, bounding boxes, confidence, and timings.
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[OCRResult]:
        """Run batch OCR inference over multiple images.

        Args:
            images: List of input images.
            request_id: Optional trace request ID.

        Returns:
            List of OCRResult instances corresponding to inputs.
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Warm up the underlying model graphs with a dummy inference pass."""
        pass

    @property
    def info(self) -> Dict[str, Any]:
        """Metadata about the engine configuration, threads, and hardware provider."""
        return {
            "name": self.name,
            "backend": "cpu",
        }
