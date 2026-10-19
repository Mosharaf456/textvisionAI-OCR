"""Document binarization, shadow elimination, and contrast enhancement."""

from __future__ import annotations

import cv2
import numpy as np


def remove_shadows(image: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """Eliminates gradient shadows, camera flash glare, and uneven illumination.

    Estimates the background illumination using morphological dilation and normalizes.
    """
    if image.ndim == 3:
        # Process in LAB or HSV color space to only affect luminance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_norm = _normalize_plane(l, kernel_size)
        lab_norm = cv2.merge((l_norm, a, b))
        return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    else:
        return _normalize_plane(image, kernel_size)


def _normalize_plane(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Internal helper to normalize illumination on a 1-channel image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Dilate to get background estimate
    bg = cv2.dilate(gray, kernel)
    # Median blur background to smooth transitions
    bg = cv2.medianBlur(bg, 21)
    # Difference from background
    diff = 255 - cv2.absdiff(gray, bg)
    # Normalize to full 0-255 dynamic range
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization for CCTV or faint scans."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def sauvola_threshold(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
) -> np.ndarray:
    """Sauvola adaptive binarization, optimal for degraded historical and printed documents.

    Threshold formula: T(x, y) = m(x, y) * [1 + k * (s(x, y) / r - 1)]
    Implemented efficiently using cv2.boxFilter.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)

    # Local mean m(x, y)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window_size, window_size), borderType=cv2.BORDER_REFLECT)

    # Local variance s(x, y) = sqrt(E[x^2] - (E[x])^2)
    sqr_mean = cv2.boxFilter(gray_f ** 2, ddepth=-1, ksize=(window_size, window_size), borderType=cv2.BORDER_REFLECT)
    var = sqr_mean - mean ** 2
    var[var < 0] = 0
    std = np.sqrt(var)

    # Sauvola threshold map
    threshold = mean * (1.0 + k * (std / r - 1.0))

    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[gray_f > threshold] = 255
    return binary


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Standard global Otsu binarization with Gaussian denoise pre-filter."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
