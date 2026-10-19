"""Recursive XY-Cut layout analysis for multi-column document reading order reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from source.core.types import TextLine


@dataclass
class LayoutBlock:
    """A semantic region containing ordered text lines."""
    block_id: int
    lines: List[TextLine]
    bbox_xywh: Tuple[int, int, int, int]


class RecursiveXYCut:
    """Partitions document bounding boxes into logical reading order using projection profiles."""

    def __init__(self, min_gap_x: float = 20.0, min_gap_y: float = 12.0):
        self.min_gap_x = min_gap_x
        self.min_gap_y = min_gap_y

    def sort_reading_order(self, lines: List[TextLine]) -> List[TextLine]:
        """Reorder text lines according to multi-column reading flow."""
        if len(lines) <= 1:
            return lines

        blocks = self._recursive_cut(lines, axis=0, block_counter=[0])
        ordered_lines: List[TextLine] = []

        for blk in blocks:
            # Within each terminal block, sort top-to-bottom, then left-to-right
            sorted_block_lines = sorted(blk.lines, key=lambda l: (l.bbox.y_min, l.bbox.x_min))
            for l in sorted_block_lines:
                l.block_id = blk.block_id
                ordered_lines.append(l)

        # Update line_ids sequentially
        for idx, line in enumerate(ordered_lines):
            line.line_id = idx

        return ordered_lines

    def _recursive_cut(
        self,
        lines: List[TextLine],
        axis: int,  # 0 = vertical cut (split X columns), 1 = horizontal cut (split Y paragraphs)
        block_counter: List[int],
    ) -> List[LayoutBlock]:
        if not lines:
            return []

        # Find cuts along current axis
        cuts = self._find_cuts(lines, axis)

        if not cuts:
            # Try alternating axis
            other_axis = 1 - axis
            cuts = self._find_cuts(lines, other_axis)
            if not cuts:
                # Terminal block reached
                blk_id = block_counter[0]
                block_counter[0] += 1
                x_min = min(l.bbox.x_min for l in lines)
                y_min = min(l.bbox.y_min for l in lines)
                w = max(l.bbox.x_max for l in lines) - x_min
                h = max(l.bbox.y_max for l in lines) - y_min
                return [LayoutBlock(block_id=blk_id, lines=lines, bbox_xywh=(int(x_min), int(y_min), int(w), int(h)))]
            axis = other_axis

        # Partition lines into groups based on cuts
        groups = self._partition_lines(lines, cuts, axis)

        # Recursively cut each group using the alternate axis
        blocks: List[LayoutBlock] = []
        next_axis = 1 - axis
        for grp in groups:
            blocks.extend(self._recursive_cut(grp, next_axis, block_counter))

        return blocks

    def _find_cuts(self, lines: List[TextLine], axis: int) -> List[float]:
        """Find valley gaps in projection histogram."""
        if axis == 0:
            # X projection (looking for vertical gutters between columns)
            intervals = [(l.bbox.x_min, l.bbox.x_max) for l in lines]
            min_gap = self.min_gap_x
        else:
            # Y projection (looking for horizontal gaps between paragraphs/sections)
            intervals = [(l.bbox.y_min, l.bbox.y_max) for l in lines]
            min_gap = self.min_gap_y

        intervals.sort(key=lambda item: item[0])

        # Merge overlapping intervals
        merged: List[Tuple[float, float]] = []
        for start, end in intervals:
            if not merged:
                merged.append((start, end))
            else:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    merged[-1] = (last_start, max(last_end, end))
                else:
                    merged.append((start, end))

        cuts: List[float] = []
        for i in range(len(merged) - 1):
            gap = merged[i + 1][0] - merged[i][1]
            if gap >= min_gap:
                cut_point = (merged[i][1] + merged[i + 1][0]) / 2.0
                cuts.append(cut_point)

        return cuts

    def _partition_lines(
        self, lines: List[TextLine], cuts: List[float], axis: int
    ) -> List[List[TextLine]]:
        """Group lines according to split boundaries."""
        groups: List[List[TextLine]] = [[] for _ in range(len(cuts) + 1)]

        for line in lines:
            center_val = line.bbox.center[0] if axis == 0 else line.bbox.center[1]
            idx = 0
            while idx < len(cuts) and center_val > cuts[idx]:
                idx += 1
            groups[idx].append(line)

        # Remove empty groups
        return [g for g in groups if g]
