"""Grid Differencing Engine: analyze input→output transformations at pixel level.

Extracts structured descriptions of what changes between input and output:
- Which cells change color
- What spatial pattern do the changes follow
- Are changes relative to objects or absolute positions

This feeds the Type Classifier with ground-truth supervision signals
and enables the rule engine to learn from demo pairs at test time.
"""

import torch
from dataclasses import dataclass
from typing import Optional

from ..data.arc_dataset import ARCPair, ARCTask, NUM_COLORS


@dataclass
class GridDiff:
    """Structured diff between an input and output grid."""
    # Basic stats
    same_shape: bool
    num_changed: int           # cells that changed color
    num_total: int             # total cells in output
    change_fraction: float     # fraction of cells that changed

    # Color analysis
    colors_added: set[int]     # colors in output but not input
    colors_removed: set[int]   # colors in input but not output
    color_mapping: dict[int, set[int]]  # for each input color, what output colors it maps to

    # Spatial analysis
    changes_mask: torch.Tensor  # (H, W) bool — which cells changed (if same shape)
    change_pattern: str         # "scattered", "contiguous", "row", "column", "border", "fill"

    # Size relationship
    h_ratio: float
    w_ratio: float


def compute_diff(pair: ARCPair) -> GridDiff:
    """Compute structured diff between input and output."""
    inp, out = pair.input, pair.output
    ih, iw = inp.shape
    oh, ow = out.shape
    same_shape = (ih == oh and iw == ow)

    # Color sets
    in_colors = set(inp.unique().tolist())
    out_colors = set(out.unique().tolist())
    colors_added = out_colors - in_colors
    colors_removed = in_colors - out_colors

    # Color mapping (only if same shape)
    color_mapping: dict[int, set[int]] = {}
    changes_mask = torch.zeros(oh, ow, dtype=torch.bool)

    if same_shape:
        changes_mask = inp != out
        num_changed = changes_mask.sum().item()

        for color in in_colors:
            mask = inp == color
            mapped_colors = out[mask].unique().tolist()
            color_mapping[int(color)] = set(mapped_colors)
    else:
        num_changed = oh * ow  # can't compare directly

    num_total = oh * ow
    change_fraction = num_changed / max(num_total, 1)

    # Classify change pattern
    change_pattern = _classify_pattern(changes_mask, oh, ow)

    return GridDiff(
        same_shape=same_shape,
        num_changed=num_changed,
        num_total=num_total,
        change_fraction=change_fraction,
        colors_added=colors_added,
        colors_removed=colors_removed,
        color_mapping=color_mapping,
        changes_mask=changes_mask,
        change_pattern=change_pattern,
        h_ratio=oh / ih,
        w_ratio=ow / iw,
    )


def _classify_pattern(mask: torch.Tensor, h: int, w: int) -> str:
    """Classify the spatial pattern of changes."""
    if not mask.any():
        return "none"

    n_changed = mask.sum().item()
    total = h * w

    if n_changed == total:
        return "full"

    if n_changed / total < 0.05:
        return "scattered"

    # Check if changes are along border
    border = torch.zeros_like(mask)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if (mask & border).sum() == n_changed:
        return "border"

    # Check if changes are in complete rows
    row_changed = mask.any(dim=1)
    row_full = mask.all(dim=1)
    if row_changed.sum() == row_full.sum() and row_full.any():
        return "row"

    # Check if changes are in complete columns
    col_changed = mask.any(dim=0)
    col_full = mask.all(dim=0)
    if col_changed.sum() == col_full.sum() and col_full.any():
        return "column"

    # Check contiguity via bounding box coverage
    if mask.any():
        rows = mask.any(dim=1).nonzero(as_tuple=True)[0]
        cols = mask.any(dim=0).nonzero(as_tuple=True)[0]
        bbox_area = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
        if n_changed / bbox_area.item() > 0.7:
            return "contiguous"

    # Check if it's a fill (all zeros become non-zero or vice versa)
    return "scattered"


def analyze_task(task: ARCTask) -> dict:
    """Analyze all demo pairs in a task to find consistent patterns.

    Returns a summary dict describing the task's transformation.
    """
    diffs = [compute_diff(pair) for pair in task.train]

    if not diffs:
        return {}

    # Check consistency across demos
    all_same_shape = all(d.same_shape for d in diffs)
    patterns = [d.change_pattern for d in diffs]
    consistent_pattern = len(set(patterns)) == 1

    h_ratios = [d.h_ratio for d in diffs]
    w_ratios = [d.w_ratio for d in diffs]
    consistent_ratio = (len(set(h_ratios)) == 1 and len(set(w_ratios)) == 1)

    # Color analysis across demos
    all_added = set.intersection(*[d.colors_added for d in diffs]) if diffs else set()
    all_removed = set.intersection(*[d.colors_removed for d in diffs]) if diffs else set()

    avg_change = sum(d.change_fraction for d in diffs) / len(diffs)

    return {
        "num_demos": len(diffs),
        "same_shape": all_same_shape,
        "consistent_pattern": consistent_pattern,
        "pattern": patterns[0] if consistent_pattern else "mixed",
        "consistent_ratio": consistent_ratio,
        "h_ratio": h_ratios[0] if consistent_ratio else None,
        "w_ratio": w_ratios[0] if consistent_ratio else None,
        "avg_change_fraction": avg_change,
        "colors_consistently_added": all_added,
        "colors_consistently_removed": all_removed,
        "diffs": diffs,
    }


def suggest_approach(analysis: dict) -> str:
    """Suggest which solving approach to use based on task analysis.

    Returns one of: "rule", "neural_easy", "neural_hard", "ttt_required"
    """
    if not analysis:
        return "ttt_required"

    # Same shape + low change fraction → likely simple rule
    if analysis["same_shape"] and analysis["avg_change_fraction"] < 0.3:
        return "rule"

    # Consistent integer scaling → tiling/scaling rule
    if analysis["consistent_ratio"]:
        hr = analysis["h_ratio"]
        wr = analysis["w_ratio"]
        if hr and wr and hr == int(hr) and wr == int(wr):
            return "rule"

    # Same shape + consistent pattern → neural should handle
    if analysis["same_shape"] and analysis["consistent_pattern"]:
        return "neural_easy"

    # Different shapes but consistent ratio → neural with size inference
    if analysis["consistent_ratio"]:
        return "neural_easy"

    # Everything else → hard, needs TTT
    return "ttt_required"
