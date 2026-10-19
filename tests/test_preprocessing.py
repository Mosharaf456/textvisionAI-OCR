"""Unit tests for document preprocessing, deskewing, binarization, and layout analysis."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from source.core.types import BoundingBox, TextLine
from source.preprocessing.binarization import apply_clahe, otsu_threshold, remove_shadows, sauvola_threshold
from source.preprocessing.deskew import Deskewer
from source.preprocessing.layout import RecursiveXYCut
from source.preprocessing.perspective import four_point_transform, order_points


def test_order_points():
    # Points in random order
    pts = np.array([[200, 200], [0, 200], [200, 0], [0, 0]], dtype=np.float32)
    ordered = order_points(pts)

    assert np.allclose(ordered[0], [0, 0])      # Top-left
    assert np.allclose(ordered[1], [200, 0])    # Top-right
    assert np.allclose(ordered[2], [200, 200])  # Bottom-right
    assert np.allclose(ordered[3], [0, 200])    # Bottom-left


def test_four_point_transform():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), -1)

    pts = [[50, 50], [250, 50], [250, 250], [50, 250]]
    warped = four_point_transform(img, pts)

    assert warped.shape[0] == 200 or warped.shape[0] == 201
    assert warped.shape[1] == 200 or warped.shape[1] == 201


def test_binarization_and_shadow_removal():
    img = np.full((100, 100), 200, dtype=np.uint8)
    # Add fake gradient shadow
    for y in range(100):
        img[y, :] = max(0, 200 - y * 2)

    no_shadow = remove_shadows(img)
    assert no_shadow.shape == (100, 100)

    binary = sauvola_threshold(no_shadow)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})

    clahe_img = apply_clahe(img)
    assert clahe_img.shape == (100, 100)


def test_recursive_xy_cut_multi_column():
    """Verify that Recursive XY-Cut sorts column 1 completely before column 2."""
    xy_cut = RecursiveXYCut(min_gap_x=30.0, min_gap_y=10.0)

    # Column 1 lines: X in [10, 100], Y at 10, 40, 70
    col1_l1 = TextLine("Col 1 Line 1", 1.0, BoundingBox.from_xywh(10, 10, 90, 20))
    col1_l2 = TextLine("Col 1 Line 2", 1.0, BoundingBox.from_xywh(10, 40, 90, 20))
    col1_l3 = TextLine("Col 1 Line 3", 1.0, BoundingBox.from_xywh(10, 70, 90, 20))

    # Column 2 lines: X in [200, 290], Y at 10, 40, 70 (gutter is 100px)
    col2_l1 = TextLine("Col 2 Line 1", 1.0, BoundingBox.from_xywh(200, 10, 90, 20))
    col2_l2 = TextLine("Col 2 Line 2", 1.0, BoundingBox.from_xywh(200, 40, 90, 20))
    col2_l3 = TextLine("Col 2 Line 3", 1.0, BoundingBox.from_xywh(200, 70, 90, 20))

    # Input lines in naive top-to-bottom order (interleaved)
    interleaved_input = [col1_l1, col2_l1, col1_l2, col2_l2, col1_l3, col2_l3]

    sorted_lines = xy_cut.sort_reading_order(interleaved_input)
    sorted_texts = [l.text for l in sorted_lines]

    expected = [
        "Col 1 Line 1",
        "Col 1 Line 2",
        "Col 1 Line 3",
        "Col 2 Line 1",
        "Col 2 Line 2",
        "Col 2 Line 3",
    ]
    assert sorted_texts == expected, f"Expected {expected}, got {sorted_texts}"
