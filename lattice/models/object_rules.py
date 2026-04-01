"""Object-level rules for ARC tasks.

These rules operate on detected objects (connected components) rather than
raw pixels. They handle patterns like:
- Copy/move objects based on color
- Fill enclosed regions
- Mirror objects across axes
- Remove/keep objects by size, color, or position
- Replace objects based on pattern matching

This is where most of the "easy but not trivial" ARC tasks live.
"""

import torch
from typing import Optional
from collections import Counter

from ..data.arc_dataset import ARCTask, ARCPair, NUM_COLORS
from .rule_engine import Rule


# --- Object extraction ---

def extract_objects(
    grid: torch.Tensor, background: int = 0, connectivity: int = 4,
) -> list[dict]:
    """Extract connected components (objects) from grid.

    Returns list of dicts with:
        cells: list of (row, col) positions
        color: int (single color) or None (multicolor)
        bbox: (r_min, r_max, c_min, c_max)
        mask: (H, W) bool tensor
        size: int (number of cells)
    """
    h, w = grid.shape
    visited = torch.zeros(h, w, dtype=torch.bool)
    objects = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] != background and not visited[r, c]:
                # BFS flood fill
                cells = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if visited[cr, cc] or grid[cr, cc] == background:
                        continue
                    visited[cr, cc] = True
                    cells.append((cr, cc))
                    if connectivity == 4:
                        stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                    else:  # 8-connected
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                stack.append((cr+dr, cc+dc))

                rows = [p[0] for p in cells]
                cols = [p[1] for p in cells]
                colors = set(grid[p[0], p[1]].item() for p in cells)
                mask = torch.zeros(h, w, dtype=torch.bool)
                for p in cells:
                    mask[p[0], p[1]] = True

                objects.append({
                    "cells": cells,
                    "color": list(colors)[0] if len(colors) == 1 else None,
                    "colors": colors,
                    "bbox": (min(rows), max(rows), min(cols), max(cols)),
                    "mask": mask,
                    "size": len(cells),
                })

    return objects


def extract_object_grid(grid: torch.Tensor, obj: dict) -> torch.Tensor:
    """Extract an object as a cropped grid (background = 0)."""
    r_min, r_max, c_min, c_max = obj["bbox"]
    cropped = torch.zeros(r_max - r_min + 1, c_max - c_min + 1, dtype=grid.dtype)
    for r, c in obj["cells"]:
        cropped[r - r_min, c - c_min] = grid[r, c]
    return cropped


# --- Object-level rule detectors ---

def detect_remove_color(task: ARCTask) -> Optional[Rule]:
    """Detect if all cells of a specific color are removed (set to 0)."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    # Find cells that changed to 0
    became_zero = (pair.input != 0) & (pair.output == 0)
    if not became_zero.any():
        return None

    # All removed cells should be same color
    removed_colors = pair.input[became_zero].unique()
    if removed_colors.numel() != 1:
        return None
    removed_color = removed_colors[0].item()

    # Everything else should be unchanged
    kept = ~became_zero
    if not torch.equal(pair.input[kept], pair.output[kept]):
        return None

    def remove_fn(grid):
        result = grid.clone()
        result[grid == removed_color] = 0
        return result

    rule = Rule(f"remove_color_{removed_color}", remove_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_keep_largest(task: ARCTask) -> Optional[Rule]:
    """Detect if output keeps only the largest object."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    in_objs = extract_objects(pair.input)
    if len(in_objs) < 2:
        return None

    largest = max(in_objs, key=lambda o: o["size"])

    # Check if output == only the largest object
    expected = torch.zeros_like(pair.input)
    for r, c in largest["cells"]:
        expected[r, c] = pair.input[r, c]

    if not torch.equal(expected, pair.output):
        return None

    def keep_largest_fn(grid):
        objs = extract_objects(grid)
        if not objs:
            return grid
        largest_obj = max(objs, key=lambda o: o["size"])
        result = torch.zeros_like(grid)
        for r, c in largest_obj["cells"]:
            result[r, c] = grid[r, c]
        return result

    rule = Rule("keep_largest_object", keep_largest_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_keep_smallest(task: ARCTask) -> Optional[Rule]:
    """Detect if output keeps only the smallest object."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    in_objs = extract_objects(pair.input)
    if len(in_objs) < 2:
        return None

    smallest = min(in_objs, key=lambda o: o["size"])

    expected = torch.zeros_like(pair.input)
    for r, c in smallest["cells"]:
        expected[r, c] = pair.input[r, c]

    if not torch.equal(expected, pair.output):
        return None

    def keep_smallest_fn(grid):
        objs = extract_objects(grid)
        if not objs:
            return grid
        smallest_obj = min(objs, key=lambda o: o["size"])
        result = torch.zeros_like(grid)
        for r, c in smallest_obj["cells"]:
            result[r, c] = grid[r, c]
        return result

    rule = Rule("keep_smallest_object", keep_smallest_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_flood_fill_enclosed(task: ARCTask) -> Optional[Rule]:
    """Detect if enclosed regions (holes) are filled with a color."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    h, w = pair.input.shape

    # Find cells that changed from 0 to non-zero
    filled = (pair.input == 0) & (pair.output != 0)
    if not filled.any():
        return None

    fill_colors = pair.output[filled].unique()
    if fill_colors.numel() != 1:
        return None
    fill_color = fill_colors[0].item()

    # Check: are the filled cells "enclosed" (not reachable from border)?
    def find_reachable_from_border(grid, bg=0):
        """BFS from all border background cells."""
        reachable = torch.zeros_like(grid, dtype=torch.bool)
        stack = []
        for r in range(h):
            for c in [0, w-1]:
                if grid[r, c] == bg:
                    stack.append((r, c))
        for c in range(w):
            for r in [0, h-1]:
                if grid[r, c] == bg:
                    stack.append((r, c))

        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= h or cc < 0 or cc >= w:
                continue
            if reachable[cr, cc] or grid[cr, cc] != bg:
                continue
            reachable[cr, cc] = True
            stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
        return reachable

    reachable = find_reachable_from_border(pair.input)
    enclosed = (pair.input == 0) & ~reachable

    # Check if filled cells match enclosed cells
    if not torch.equal(filled, enclosed):
        return None

    def flood_fill_fn(grid):
        gh, gw = grid.shape
        reachable_bg = torch.zeros_like(grid, dtype=torch.bool)
        stack = []
        for r in range(gh):
            for c in [0, gw-1]:
                if grid[r, c] == 0:
                    stack.append((r, c))
        for c in range(gw):
            for r in [0, gh-1]:
                if grid[r, c] == 0:
                    stack.append((r, c))
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= gh or cc < 0 or cc >= gw:
                continue
            if reachable_bg[cr, cc] or grid[cr, cc] != 0:
                continue
            reachable_bg[cr, cc] = True
            stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])

        result = grid.clone()
        enclosed_cells = (grid == 0) & ~reachable_bg
        result[enclosed_cells] = fill_color
        return result

    rule = Rule(f"flood_fill_enclosed_{fill_color}", flood_fill_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_object_count_output(task: ARCTask) -> Optional[Rule]:
    """Detect if output is a small grid encoding the count of objects/colors."""
    pair = task.train[0]
    oh, ow = pair.output.shape

    # Output should be small (1x1, 1xN, Nx1)
    if oh > 3 or ow > 3:
        return None

    in_objs = extract_objects(pair.input)
    n_objs = len(in_objs)

    # Check if output encodes object count
    if oh == 1 and ow == 1:
        out_val = pair.output[0, 0].item()
        if out_val == n_objs:
            def count_fn(grid):
                objs = extract_objects(grid)
                return torch.tensor([[len(objs)]], dtype=grid.dtype)
            rule = Rule("count_objects", count_fn)
            if rule.verify(task.train):
                return rule

    return None


def detect_recolor_by_size(task: ARCTask) -> Optional[Rule]:
    """Detect if objects are recolored based on their size."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    in_objs = extract_objects(pair.input)
    out_objs = extract_objects(pair.output)

    if len(in_objs) != len(out_objs) or len(in_objs) < 2:
        return None

    # Match objects by position (same bbox)
    size_to_color = {}
    for in_obj in in_objs:
        # Find matching output object
        for out_obj in out_objs:
            if in_obj["bbox"] == out_obj["bbox"]:
                if out_obj["color"] is not None:
                    size_to_color[in_obj["size"]] = out_obj["color"]
                break

    if len(size_to_color) < 2:
        return None

    def recolor_fn(grid):
        result = grid.clone()
        objs = extract_objects(grid)
        for obj in objs:
            if obj["size"] in size_to_color:
                new_color = size_to_color[obj["size"]]
                for r, c in obj["cells"]:
                    result[r, c] = new_color
        return result

    rule = Rule("recolor_by_size", recolor_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_mirror_object(task: ARCTask) -> Optional[Rule]:
    """Detect if a single object is mirrored/reflected within the grid."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    h, w = pair.input.shape

    # Check horizontal mirror: output = input reflected left-right
    # But only non-background cells get mirrored
    for axis in ["h", "v"]:
        def make_mirror(ax):
            def mirror_fn(grid):
                gh, gw = grid.shape
                result = grid.clone()
                if ax == "h":
                    # Mirror non-zero cells horizontally
                    for r in range(gh):
                        for c in range(gw):
                            if grid[r, c] != 0:
                                mirror_c = gw - 1 - c
                                result[r, mirror_c] = grid[r, c]
                else:
                    for r in range(gh):
                        for c in range(gw):
                            if grid[r, c] != 0:
                                mirror_r = gh - 1 - r
                                result[mirror_r, c] = grid[r, c]
                return result
            return mirror_fn

        rule = Rule(f"mirror_objects_{axis}", make_mirror(axis))
        if rule.verify(task.train):
            return rule

    return None


# --- Aggregate ---

OBJECT_DETECTORS = [
    detect_remove_color,
    detect_keep_largest,
    detect_keep_smallest,
    detect_flood_fill_enclosed,
    detect_object_count_output,
    detect_recolor_by_size,
    detect_mirror_object,
]
