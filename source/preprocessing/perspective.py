"""Perspective transformation and 4-point homography rectification."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order coordinates in consistent sequence: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)

    # top-left has smallest sum (x + y), bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right has smallest diff (y - x), bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
    """Warp quadrilateral region defined by 4 points into a top-down planar rectangle."""
    pts_arr = np.array(pts, dtype=np.float32)
    rect = order_points(pts_arr)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(round(width_a)), int(round(width_b)))

    # Compute height of new image
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(round(height_a)), int(round(height_b)))

    if max_width <= 0 or max_height <= 0:
        return image

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Compute perspective transform matrix and warp
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height), flags=cv2.INTER_CUBIC)
    return warped


def find_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Locate 4-point bounding polygon of a card, paper, or license plate in an image."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    h, w = gray.shape[:2]
    image_area = w * h

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (image_area * 0.05):
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None
