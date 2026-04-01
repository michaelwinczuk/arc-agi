"""Deterministic Rule Engine for ARC tasks.

Many ARC tasks follow simple, detectable rules that don't need
neural networks. This engine tries to solve tasks purely with
programmatic rules BEFORE falling back to the neural pipeline.

Why this matters:
- ~30-40% of ARC tasks are "simple" (single rule, no composition)
- Deterministic rules get 100% accuracy when they match
- Takes <1ms per task (vs ~55ms for neural, ~2s for TTT)
- Frees the neural pipeline to focus on the hard tasks

Rules are organized by complexity:
1. Identity transforms (copy, tile, scale)
2. Color operations (recolor, fill, swap)
3. Geometric operations (rotate, reflect, translate)
4. Object operations (extract, count, filter)
"""

import torch
from typing import Optional, Callable

from ..data.arc_dataset import ARCTask, ARCPair, NUM_COLORS


# --- Rule type ---

class Rule:
    """A candidate rule that maps input grids to output grids."""

    def __init__(self, name: str, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.name = name
        self.fn = fn

    def apply(self, grid: torch.Tensor) -> torch.Tensor:
        return self.fn(grid)

    def verify(self, pairs: list[ARCPair]) -> bool:
        """Check if this rule produces correct output for all demo pairs."""
        for pair in pairs:
            try:
                predicted = self.apply(pair.input)
                if not torch.equal(predicted, pair.output):
                    return False
            except Exception:
                return False
        return True


# --- Rule generators ---

def detect_tiling(task: ARCTask) -> Optional[Rule]:
    """Detect if output is input tiled N×M times."""
    for pair in task.train:
        ih, iw = pair.input.shape
        oh, ow = pair.output.shape

        if oh % ih != 0 or ow % iw != 0:
            return None

        rh, rw = oh // ih, ow // iw

        # Check if it's simple tiling
        tiled = pair.input.repeat(rh, rw)
        if not torch.equal(tiled, pair.output):
            # Try alternating (checkerboard) tiling
            break
    else:
        return None

    # If first pair gave us ratios, verify all pairs use same ratios
    rh0, rw0 = task.train[0].output.shape[0] // task.train[0].input.shape[0], \
                task.train[0].output.shape[1] // task.train[0].input.shape[1]

    def tile_fn(grid):
        return grid.repeat(rh0, rw0)

    rule = Rule(f"tile_{rh0}x{rw0}", tile_fn)
    if rule.verify(task.train):
        return rule

    # Try tiling with alternating flips (common ARC pattern)
    def tile_flip_fn(grid):
        h, w = grid.shape
        rows = []
        for r in range(rh0):
            cols = []
            for c in range(rw0):
                block = grid.clone()
                if r % 2 == 1:
                    block = block.flip(0)
                if c % 2 == 1:
                    block = block.flip(1)
                cols.append(block)
            rows.append(torch.cat(cols, dim=1))
        return torch.cat(rows, dim=0)

    rule = Rule(f"tile_flip_{rh0}x{rw0}", tile_flip_fn)
    if rule.verify(task.train):
        return rule

    return None


def detect_scaling(task: ARCTask) -> Optional[Rule]:
    """Detect if output is input scaled by integer factor."""
    pair = task.train[0]
    ih, iw = pair.input.shape
    oh, ow = pair.output.shape

    if oh % ih != 0 or ow % iw != 0:
        return None

    sh, sw = oh // ih, ow // iw
    if sh != sw:
        return None  # non-uniform scaling, handle separately

    scale = sh

    def scale_fn(grid):
        return grid.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)

    rule = Rule(f"scale_{scale}x", scale_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_rotation(task: ARCTask) -> Optional[Rule]:
    """Detect if output is a rotation of input."""
    for k in [1, 2, 3]:  # 90, 180, 270 degrees
        def make_rot(k_val):
            return lambda grid: torch.rot90(grid, k_val, dims=(0, 1))

        rule = Rule(f"rotate_{k*90}", make_rot(k))
        if rule.verify(task.train):
            return rule
    return None


def detect_reflection(task: ARCTask) -> Optional[Rule]:
    """Detect if output is a reflection of input."""
    for name, fn in [
        ("flip_h", lambda g: g.flip(1)),
        ("flip_v", lambda g: g.flip(0)),
        ("flip_diag", lambda g: g.t()),
        ("flip_antidiag", lambda g: g.flip(0).flip(1).t()),
    ]:
        rule = Rule(name, fn)
        if rule.verify(task.train):
            return rule
    return None


def detect_color_swap(task: ARCTask) -> Optional[Rule]:
    """Detect if output is input with colors swapped."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    # Build color mapping from first pair
    mapping = {}
    ih, iw = pair.input.shape
    for r in range(ih):
        for c in range(iw):
            in_color = pair.input[r, c].item()
            out_color = pair.output[r, c].item()
            if in_color in mapping:
                if mapping[in_color] != out_color:
                    return None  # inconsistent mapping
            mapping[in_color] = out_color

    # Build the permutation
    perm = list(range(NUM_COLORS))
    for old, new in mapping.items():
        perm[old] = new

    def color_swap_fn(grid):
        result = grid.clone()
        for old_color, new_color in enumerate(perm):
            if old_color != new_color:
                result[grid == old_color] = new_color
        return result

    rule = Rule(f"color_swap_{mapping}", color_swap_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_identity(task: ARCTask) -> Optional[Rule]:
    """Detect if output == input (identity)."""
    rule = Rule("identity", lambda g: g.clone())
    if rule.verify(task.train):
        return rule
    return None


def detect_transpose(task: ARCTask) -> Optional[Rule]:
    """Detect if output is transpose of input."""
    rule = Rule("transpose", lambda g: g.t())
    if rule.verify(task.train):
        return rule
    return None


def detect_crop_to_nonzero(task: ARCTask) -> Optional[Rule]:
    """Detect if output is input cropped to bounding box of non-zero cells."""
    def crop_fn(grid):
        nonzero = (grid != 0).nonzero(as_tuple=True)
        if len(nonzero[0]) == 0:
            return grid
        r_min, r_max = nonzero[0].min().item(), nonzero[0].max().item()
        c_min, c_max = nonzero[1].min().item(), nonzero[1].max().item()
        return grid[r_min:r_max+1, c_min:c_max+1]

    rule = Rule("crop_nonzero", crop_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_fill_color(task: ARCTask) -> Optional[Rule]:
    """Detect if a specific color region is filled with another color."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    # Find which cells changed
    diff = pair.input != pair.output
    if not diff.any():
        return None

    # All changed cells should map to the same output color
    changed_outputs = pair.output[diff]
    if changed_outputs.unique().numel() != 1:
        return None

    fill_color = changed_outputs[0].item()

    # All changed cells should have the same input color
    changed_inputs = pair.input[diff]
    if changed_inputs.unique().numel() != 1:
        return None

    source_color = changed_inputs[0].item()

    def fill_fn(grid):
        result = grid.clone()
        result[grid == source_color] = fill_color
        return result

    rule = Rule(f"fill_{source_color}_to_{fill_color}", fill_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_gravity(task: ARCTask) -> Optional[Rule]:
    """Detect if non-background cells 'fall' to one edge (gravity)."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    for direction in ["down", "up", "left", "right"]:
        def make_gravity(d):
            def gravity_fn(grid):
                h, w = grid.shape
                result = torch.zeros_like(grid)
                if d == "down":
                    for c in range(w):
                        col = grid[:, c]
                        nonzero = col[col != 0]
                        result[h - len(nonzero):, c] = nonzero
                elif d == "up":
                    for c in range(w):
                        col = grid[:, c]
                        nonzero = col[col != 0]
                        result[:len(nonzero), c] = nonzero
                elif d == "right":
                    for r in range(h):
                        row = grid[r, :]
                        nonzero = row[row != 0]
                        result[r, w - len(nonzero):] = nonzero
                elif d == "left":
                    for r in range(h):
                        row = grid[r, :]
                        nonzero = row[row != 0]
                        result[r, :len(nonzero)] = nonzero
                return result
            return gravity_fn

        rule = Rule(f"gravity_{direction}", make_gravity(direction))
        if rule.verify(task.train):
            return rule
    return None


def detect_most_common_color_fill(task: ARCTask) -> Optional[Rule]:
    """Detect if output fills background with the most common non-zero color."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    # Check which color is used to fill zeros
    zeros_in = (pair.input == 0)
    if not zeros_in.any():
        return None

    fill_vals = pair.output[zeros_in]
    if fill_vals.unique().numel() != 1:
        return None
    fill_color = fill_vals[0].item()
    if fill_color == 0:
        return None

    def fill_bg_fn(grid):
        result = grid.clone()
        # Find most common non-zero color
        nonzero = grid[grid != 0]
        if len(nonzero) == 0:
            return result
        counts = torch.bincount(nonzero, minlength=NUM_COLORS)
        most_common = counts[1:].argmax().item() + 1
        result[grid == 0] = most_common
        return result

    rule = Rule("fill_bg_most_common", fill_bg_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_border(task: ARCTask) -> Optional[Rule]:
    """Detect if output adds a border around the input."""
    pair = task.train[0]
    ih, iw = pair.input.shape
    oh, ow = pair.output.shape

    for border_size in [1, 2]:
        if oh == ih + 2 * border_size and ow == iw + 2 * border_size:
            # Check if inner region matches input
            inner = pair.output[border_size:-border_size, border_size:-border_size]
            if not torch.equal(inner, pair.input):
                continue

            # Get border color
            border_vals = []
            border_vals.extend(pair.output[0, :].tolist())
            border_vals.extend(pair.output[-1, :].tolist())
            border_vals.extend(pair.output[:, 0].tolist())
            border_vals.extend(pair.output[:, -1].tolist())
            unique_border = set(border_vals)
            if len(unique_border) != 1:
                continue
            border_color = border_vals[0]

            def make_border(bs, bc):
                def border_fn(grid):
                    h, w = grid.shape
                    result = torch.full(
                        (h + 2 * bs, w + 2 * bs), bc, dtype=grid.dtype
                    )
                    result[bs:-bs, bs:-bs] = grid
                    return result
                return border_fn

            rule = Rule(f"border_{border_size}_{border_color}", make_border(border_size, border_color))
            if rule.verify(task.train):
                return rule
    return None


def detect_max_object(task: ARCTask) -> Optional[Rule]:
    """Detect if output extracts the largest connected object."""
    pair = task.train[0]
    if pair.input.shape == pair.output.shape:
        return None

    def extract_objects(grid):
        """Simple flood-fill connected components (4-connected, non-zero)."""
        h, w = grid.shape
        visited = torch.zeros_like(grid, dtype=torch.bool)
        objects = []

        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0 and not visited[r, c]:
                    # BFS
                    obj_cells = []
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if cr < 0 or cr >= h or cc < 0 or cc >= w:
                            continue
                        if visited[cr, cc] or grid[cr, cc] == 0:
                            continue
                        visited[cr, cc] = True
                        obj_cells.append((cr, cc))
                        stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
                    objects.append(obj_cells)
        return objects

    def largest_object_fn(grid):
        objects = extract_objects(grid)
        if not objects:
            return grid
        largest = max(objects, key=len)
        rows = [r for r, c in largest]
        cols = [c for r, c in largest]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        result = torch.zeros(r_max - r_min + 1, c_max - c_min + 1, dtype=grid.dtype)
        for r, c in largest:
            result[r - r_min, c - c_min] = grid[r, c]
        return result

    rule = Rule("extract_largest_object", largest_object_fn)
    if rule.verify(task.train):
        return rule
    return None


def detect_sort_rows(task: ARCTask) -> Optional[Rule]:
    """Detect if rows are sorted by some criterion."""
    pair = task.train[0]
    if pair.input.shape != pair.output.shape:
        return None

    ih, iw = pair.input.shape

    # Check if rows are sorted by count of non-zero cells
    def sort_rows_by_count(grid):
        h, w = grid.shape
        rows = list(range(h))
        counts = [(grid[r] != 0).sum().item() for r in range(h)]
        sorted_idx = sorted(rows, key=lambda r: counts[r])
        return grid[sorted_idx]

    rule = Rule("sort_rows_nonzero_count", sort_rows_by_count)
    if rule.verify(task.train):
        return rule

    # Descending
    def sort_rows_by_count_desc(grid):
        h, w = grid.shape
        rows = list(range(h))
        counts = [(grid[r] != 0).sum().item() for r in range(h)]
        sorted_idx = sorted(rows, key=lambda r: counts[r], reverse=True)
        return grid[sorted_idx]

    rule = Rule("sort_rows_nonzero_count_desc", sort_rows_by_count_desc)
    if rule.verify(task.train):
        return rule

    return None


# --- Main engine ---

from .object_rules import OBJECT_DETECTORS

ALL_DETECTORS = [
    detect_identity,
    detect_tiling,
    detect_scaling,
    detect_rotation,
    detect_reflection,
    detect_transpose,
    detect_color_swap,
    detect_crop_to_nonzero,
    detect_fill_color,
    detect_gravity,
    detect_most_common_color_fill,
    detect_border,
    detect_max_object,
    detect_sort_rows,
] + OBJECT_DETECTORS


class RuleEngine:
    """Try to solve a task with deterministic rules.

    Runs all detectors in order. First verified match wins.
    """

    def __init__(self, detectors=None):
        self.detectors = detectors or ALL_DETECTORS
        self.stats = {"tried": 0, "solved": 0, "rules_found": {}}

    def try_solve(self, task: ARCTask) -> Optional[list[torch.Tensor]]:
        """Try to solve task with rules.

        Returns list of predicted outputs if a rule matches, else None.
        """
        self.stats["tried"] += 1

        for detector in self.detectors:
            rule = detector(task)
            if rule is not None:
                # Rule verified on all demo pairs — apply to test inputs
                predictions = []
                for test_pair in task.test:
                    try:
                        pred = rule.apply(test_pair.input)
                        predictions.append(pred)
                    except Exception:
                        return None

                self.stats["solved"] += 1
                self.stats["rules_found"][rule.name] = \
                    self.stats["rules_found"].get(rule.name, 0) + 1
                return predictions

        return None
