"""Visualization utilities for ARC grids and slot attention.

Generates matplotlib figures for debugging and analysis.
ARC color palette follows the official specification.
"""

import torch

# Official ARC color palette (RGB)
ARC_COLORS = [
    (0, 0, 0),        # 0: black
    (0, 116, 217),    # 1: blue
    (255, 65, 54),    # 2: red
    (46, 204, 64),    # 3: green
    (255, 220, 0),    # 4: yellow
    (170, 170, 170),  # 5: grey
    (240, 18, 190),   # 6: magenta
    (255, 133, 27),   # 7: orange
    (127, 219, 255),  # 8: light blue
    (135, 12, 37),    # 9: maroon
]


def grid_to_rgb(grid: torch.Tensor) -> torch.Tensor:
    """Convert (H, W) int grid to (H, W, 3) uint8 RGB tensor."""
    h, w = grid.shape
    rgb = torch.zeros(h, w, 3, dtype=torch.uint8)
    for color_idx, (r, g, b) in enumerate(ARC_COLORS):
        mask = grid == color_idx
        rgb[mask, 0] = r
        rgb[mask, 1] = g
        rgb[mask, 2] = b
    return rgb


def print_grid(grid: torch.Tensor):
    """Print a grid to terminal with color numbers."""
    h, w = grid.shape
    for row in range(h):
        print(" ".join(str(grid[row, col].item()) for col in range(w)))


def print_task(task):
    """Print all pairs in a task."""
    print(f"Task: {task.task_id}")
    print(f"  {len(task.train)} demo pairs, {len(task.test)} test inputs")
    for i, pair in enumerate(task.train):
        print(f"\n  Demo {i+1} input ({pair.input.shape[0]}x{pair.input.shape[1]}):")
        print_grid(pair.input)
        print(f"  Demo {i+1} output ({pair.output.shape[0]}x{pair.output.shape[1]}):")
        print_grid(pair.output)
    for i, pair in enumerate(task.test):
        print(f"\n  Test {i+1} input ({pair.input.shape[0]}x{pair.input.shape[1]}):")
        print_grid(pair.input)
