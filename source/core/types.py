"""Core data structures and types for CPU OCR pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Point:
    """2D spatial coordinate."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))


@dataclass
class BoundingBox:
    """Four-point oriented polygon bounding box."""
    points: List[Point]  # [top-left, top-right, bottom-right, bottom-left]

    @classmethod
    def from_polygon(cls, polygon: Any) -> "BoundingBox":
        """Construct from list or numpy array of 4 coordinate pairs."""
        pts = []
        for pt in polygon:
            if isinstance(pt, (list, tuple, np.ndarray)):
                pts.append(Point(x=float(pt[0]), y=float(pt[1])))
            elif isinstance(pt, Point):
                pts.append(pt)
            else:
                raise ValueError(f"Invalid point format: {pt}")
        if len(pts) != 4:
            raise ValueError(f"BoundingBox requires exactly 4 points, got {len(pts)}")
        return cls(points=pts)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        """Construct axis-aligned bounding box from x, y, width, height."""
        return cls(points=[
            Point(x=x, y=y),
            Point(x=x + w, y=y),
            Point(x=x + w, y=y + h),
            Point(x=x, y=y + h)
        ])

    @property
    def x_min(self) -> float:
        return min(p.x for p in self.points)

    @property
    def x_max(self) -> float:
        return max(p.x for p in self.points)

    @property
    def y_min(self) -> float:
        return min(p.y for p in self.points)

    @property
    def y_max(self) -> float:
        return max(p.y for p in self.points)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def area(self) -> float:
        # Shoelace formula for polygon area
        n = len(self.points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i].x * self.points[j].y
            area -= self.points[j].x * self.points[i].y
        return abs(area) / 2.0

    def to_list(self) -> List[List[float]]:
        return [[p.x, p.y] for p in self.points]

    def to_int_list(self) -> List[List[int]]:
        return [[int(round(p.x)), int(round(p.y))] for p in self.points]

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.x_min)),
            int(round(self.y_min)),
            int(round(self.width)),
            int(round(self.height))
        )


@dataclass
class TextLine:
    """A recognized line of text with spatial localization and confidence."""
    text: str
    confidence: float
    bbox: BoundingBox
    angle: float = 0.0
    line_id: int = 0
    block_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_list(),
            "angle": self.angle,
            "line_id": self.line_id,
            "block_id": self.block_id,
        }


@dataclass
class OCRResult:
    """Aggregated OCR output for an image or document page."""
    lines: List[TextLine] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)  # (height, width)
    latency_ms: float = 0.0
    preprocessing_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenated text in reading order."""
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_text": self.full_text,
            "lines": [l.to_dict() for l in self.lines],
            "total_lines": len(self.lines),
            "mean_confidence": round(self.mean_confidence, 4),
            "image_shape": self.image_shape,
            "latency_ms": round(self.latency_ms, 2),
            "breakdown_ms": {
                "preprocessing": round(self.preprocessing_latency_ms, 2),
                "inference": round(self.inference_latency_ms, 2),
            },
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
