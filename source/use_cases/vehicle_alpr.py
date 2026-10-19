"""Use Case 4: Vehicle License Plate Recognition (ALPR/ANPR) for CCTV Surveillance Systems.

Features:
- Plate candidate localization & aspect-ratio filtering.
- CLAHE contrast enhancement for low-light / headlights glare.
- Regional syntax matching (Bangladesh Metro/District/Digital & International).
- Character ambiguity resolution ('0' vs 'O', '8' vs 'B').
- Temporal multi-frame majority voting for CCTV video streams.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from source.core.factory import get_ocr_engine
from source.core.logger import create_request_id, get_logger
from source.core.types import BoundingBox, OCRResult, TextLine
from source.preprocessing.binarization import apply_clahe
from source.schemas.alpr_schema import PlateBoundingBox, PlateDetection, TrackEvent
from source.schemas.plate_formats import (
    disambiguate_plate_text,
    match_plate_format,
    normalize_bangla_digits,
)

logger = get_logger("use_cases.vehicle_alpr")


class VehicleALPR:
    """Surveillance CCTV Automatic License Plate Recognition engine."""

    def __init__(
        self,
        target_region: str = "BD",
        apply_contrast_enhancement: bool = True,
    ):
        self.target_region = target_region
        self.apply_contrast_enhancement = apply_contrast_enhancement
        self.engine = get_ocr_engine()

    def process_frame(
        self,
        frame_input: Union[str, Path, np.ndarray],
        request_id: Optional[str] = None,
    ) -> List[PlateDetection]:
        """Detect and recognize license plates in a single CCTV video frame."""
        req_id = request_id or create_request_id()

        if isinstance(frame_input, (str, Path)):
            img = cv2.imread(str(frame_input))
            if img is None:
                raise ValueError(f"Could not read CCTV frame: {frame_input}")
        else:
            img = frame_input

        h, w = img.shape[:2]

        # 1. Multi-Scale Adaptive Resolution Strategy:
        # Surveillance frames or web uploads can be low resolution (e.g. 300x225).
        # DBNet text detector requires character height >= 10-15px.
        # If minimum dimension is < 700px, upscale frame for high detection sensitivity.
        scale = max(1.0, 720.0 / min(h, w)) if min(h, w) < 700 else 1.0
        if scale > 1.0:
            proc_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            proc_img = img

        # Enhance contrast if nighttime or high-contrast glare
        if self.apply_contrast_enhancement:
            enhanced_img = apply_clahe(proc_img, clip_limit=2.5)
        else:
            enhanced_img = proc_img

        ocr_result = self.engine.predict(enhanced_img, request_id=req_id)

        # Fallback: If 0 lines detected on low contrast or distant vehicles, try 2x scale
        if len(ocr_result.lines) == 0 and scale == 1.0:
            scale = 2.0
            proc_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            enhanced_img = apply_clahe(proc_img, clip_limit=3.0)
            ocr_result = self.engine.predict(enhanced_img, request_id=req_id)

        detections: List[PlateDetection] = []

        # Analyze each detected text cluster for license plate characteristics
        for line in ocr_result.lines:
            raw_text = line.text.strip()
            if len(raw_text) < 4:
                continue

            # Scale coordinates back to original image space
            lx, ly, lw, lh = line.bbox.to_xywh()
            orig_x = max(0, int(lx / scale))
            orig_y = max(0, int(ly / scale))
            orig_w = min(w - orig_x, int(lw / scale))
            orig_h = min(h - orig_y, int(lh / scale))
            aspect = orig_w / float(orig_h) if orig_h > 0 else 0.0

            # Plate geometric heuristic: plates have aspect ratio between 1.2 and 6.5
            is_plate_geometry = (1.2 <= aspect <= 6.5)

            # Two-Stage Crop Refinement:
            # Crop candidate plate region with padding to run high-res character recognition
            final_text = raw_text
            final_conf = line.confidence

            pad_x = int(orig_w * 0.10)
            pad_y = int(orig_h * 0.10)
            x1 = max(0, orig_x - pad_x)
            y1 = max(0, orig_y - pad_y)
            x2 = min(w, orig_x + orig_w + pad_x)
            y2 = min(h, orig_y + orig_h + pad_y)

            plate_crop = img[y1:y2, x1:x2]
            ch, cw = plate_crop.shape[:2]
            if ch >= 8 and cw >= 20:
                crop_scale = max(2.0, 64.0 / ch)
                hi_crop = cv2.resize(plate_crop, (int(cw * crop_scale), int(ch * crop_scale)), interpolation=cv2.INTER_CUBIC)
                crop_res = self.engine.predict(hi_crop, request_id=f"{req_id}-crop")
                if crop_res.lines and crop_res.lines[0].confidence >= 0.70:
                    best_crop_line = max(crop_res.lines, key=lambda l: l.confidence)
                    if len(best_crop_line.text.strip()) >= 4:
                        final_text = best_crop_line.text.strip()
                        final_conf = best_crop_line.confidence

            cleaned_text = disambiguate_plate_text(final_text)
            is_valid, format_name, meta = match_plate_format(cleaned_text)

            # Also test with Bangla digits normalized
            if not is_valid:
                norm_text = normalize_bangla_digits(cleaned_text)
                is_valid, format_name, meta = match_plate_format(norm_text)
                if is_valid:
                    cleaned_text = norm_text

            # Check if text qualifies as a plate candidate:
            # 1. Valid format matched from registry (BD, US, UK, Euro, Vanity)
            # 2. Or plate geometry matches + alphanumeric content of length 4-12
            is_candidate = (
                is_valid
                or (is_plate_geometry and bool(re.match(r"^[A-Z0-9\s\-–—]{4,12}$", cleaned_text, re.IGNORECASE)))
                or bool(re.search(r"\d{2}[-\s]?\d{4}", cleaned_text))
                or any(k in raw_text for k in ["মেট্রো", "METRO", "ঢাকা", "DHAKA"])
            )

            if is_candidate:
                detections.append(
                    PlateDetection(
                        plate_text=final_text,
                        normalized_text=cleaned_text,
                        confidence=round(final_conf, 4),
                        format_name=format_name or ("STANDARD_ALPHANUMERIC" if is_plate_geometry else "CUSTOM_PLATE"),
                        is_format_valid=is_valid,
                        script=meta.get("script", "latin") if meta else "latin",
                        bbox=PlateBoundingBox(x=orig_x, y=orig_y, width=orig_w, height=orig_h),
                    )
                )

        return detections


class TemporalPlateTracker:
    """Aggregates CCTV frame detections across time to eliminate transient noise via majority voting."""

    def __init__(self, min_consensus_frames: int = 3, max_history: int = 30):
        self.min_consensus_frames = min_consensus_frames
        self.max_history = max_history
        # track_id -> List of (plate_text, confidence, is_valid)
        self.tracks: Dict[int, List[Tuple[str, float, bool]]] = collections.defaultdict(list)

    def update(self, track_id: int, detection: PlateDetection) -> Optional[TrackEvent]:
        """Update tracker with detection from a new video frame."""
        history = self.tracks[track_id]
        history.append((detection.normalized_text, detection.confidence, detection.is_format_valid))

        # Keep buffer capped
        if len(history) > self.max_history:
            history.pop(0)

        # Compute consensus candidate
        readings = [item[0] for item in history]
        counter = collections.Counter(readings)

        # Prioritize valid format matches with high frequency
        most_common_plate, count = counter.most_common(1)[0]

        # Calculate average confidence for winning candidate
        confidences = [item[1] for item in history if item[0] == most_common_plate]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        is_valid, _, _ = match_plate_format(most_common_plate)

        return TrackEvent(
            track_id=track_id,
            plate_text=most_common_plate,
            confidence=round(avg_conf, 4),
            total_frames_observed=len(history),
            is_valid_format=is_valid,
            all_readings=readings,
        )

    def clear(self) -> None:
        """Reset tracking state."""
        self.tracks.clear()
