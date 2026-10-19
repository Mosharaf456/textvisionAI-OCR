"""High-precision document deskewing using contour minimum area rect and Hough transforms."""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np

from source.core.logger import get_logger

logger = get_logger("preprocessing.deskew")


class Deskewer:
    """Detects and corrects rotational skew in document images."""

    def __init__(self, max_angle: float = 45.0, min_angle_threshold: float = 0.5):
        self.max_angle = max_angle
        self.min_angle_threshold = min_angle_threshold

    def estimate_skew_angle(self, image: np.ndarray) -> float:
        """Estimate document skew angle in degrees using text contour orientations."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Invert colors if background is light: text should be white on black background
        # Otsu thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate text horizontally to connect letters into continuous word/line blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Find contours of lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        angles = []
        h, w = gray.shape[:2]
        min_area = (w * h) * 0.0005  # Filter out tiny noise

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            rect = cv2.minAreaRect(cnt)
            # rect: ((center_x, center_y), (width, height), angle)
            (cx, cy), (width, height), angle = rect

            # OpenCV minAreaRect angle normalization:
            # angle ranges from -90 to 0 (or 0 to 90 depending on OpenCV version)
            if width < height:
                angle = angle + 90.0 if angle < 0 else angle - 90.0

            # Filter out non-horizontal or extreme angles
            if abs(angle) <= self.max_angle:
                angles.append(angle)

        if not angles:
            return 0.0

        # Use median angle for robust rejection of outliers
        median_angle = float(np.median(angles))

        if abs(median_angle) < self.min_angle_threshold:
            return 0.0

        if abs(median_angle) > self.max_angle:
            logger.warning("Extreme skew detected exceeding safe threshold", angle=median_angle)
            return 0.0

        return median_angle

    def rotate_image(
        self,
        image: np.ndarray,
        angle: float,
        border_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Rotate image by angle (degrees) expanding canvas to avoid clipping corners."""
        if abs(angle) < 0.01:
            return image

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        # Calculate rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute new bounding dimensions
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take into account translation
        rot_mat[0, 2] += (new_w / 2.0) - center[0]
        rot_mat[1, 2] += (new_h / 2.0) - center[1]

        border_val = border_color if image.ndim == 3 else border_color[0]
        rotated = cv2.warpAffine(
            image,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )
        return rotated

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct skew in a single call."""
        angle = self.estimate_skew_angle(image)
        if abs(angle) >= self.min_angle_threshold:
            logger.info("Deskewing image", detected_angle=round(angle, 2))
            deskewed = self.rotate_image(image, angle)
            return deskewed, angle
        return image, 0.0
