"""Document preprocessing, deskewing, binarization, and layout analysis."""

from source.preprocessing.binarization import (
    apply_clahe,
    otsu_threshold,
    remove_shadows,
    sauvola_threshold,
)
from source.preprocessing.deskew import Deskewer
from source.preprocessing.layout import LayoutBlock, RecursiveXYCut
from source.preprocessing.perspective import (
    find_document_contour,
    four_point_transform,
    order_points,
)

__all__ = [
    "Deskewer",
    "LayoutBlock",
    "RecursiveXYCut",
    "apply_clahe",
    "find_document_contour",
    "four_point_transform",
    "order_points",
    "otsu_threshold",
    "remove_shadows",
    "sauvola_threshold",
]
