"""Pydantic schemas for CCTV vehicle tracking and license plate recognition."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class PlateBoundingBox(BaseModel):
    """Bounding coordinates of detected license plate."""
    x: int
    y: int
    width: int
    height: int


class PlateDetection(BaseModel):
    """Individual license plate reading."""
    plate_text: str = Field(..., description="Recognized plate string")
    normalized_text: str = Field(..., description="Standardized plate string without noise")
    confidence: float = Field(..., description="OCR confidence score")
    format_name: Optional[str] = Field(None, description="Matched regional format key")
    is_format_valid: bool = Field(False, description="Whether plate passed regex syntax validation")
    script: str = Field("latin", description="'bangla' or 'latin'")
    bbox: Optional[PlateBoundingBox] = None


class TrackEvent(BaseModel):
    """Consolidated vehicle tracking event across multiple CCTV frames."""
    track_id: int = Field(..., description="Unique vehicle tracking identifier")
    plate_text: str = Field(..., description="Consensus plate text after temporal majority voting")
    confidence: float = Field(..., description="Aggregated confidence score")
    total_frames_observed: int = Field(1, description="Number of frames observed")
    is_valid_format: bool = Field(False, description="Whether consensus plate matches syntax")
    all_readings: List[str] = Field(default_factory=list, description="Historical readings per frame")
