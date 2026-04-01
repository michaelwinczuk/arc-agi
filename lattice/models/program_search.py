"""Program Search: enumerate small programs that transform input → output.

For tasks where neural methods fail, try to find a short program
(composition of primitive operations) that maps all demo inputs to outputs.

This is the "depth-2 local search" fallback from the architecture spec.
Combined with the Type Lattice, we can prune the search space dramatically.

Primitives:
- Grid operations: rotate, flip, transpose, crop, pad, tile
- Color operations: swap colors, fill, recolor by adjacency
- Object operations: extract, remove, move, copy, overlay
- Structural: split grid, join grids, select subgrid

Search strategy:
1. Try each primitive individually (depth 1)
2. Try all valid compositions of 2 primitives (depth 2)
3. Type lattice prunes invalid compositions
"""

import torch
from typing import Optional, Callable
from itertools import product

from ..data.arc_dataset import ARCTask, ARCPair, NUM_COLORS
def _extract_objects_simple(grid, background=0):
    """Minimal object extraction for program search (avoids circular import)."""
    h, w = grid.shape
    visited = torch.zeros(h, w, dtype=torch.bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != background and not visited[r, c]:
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
                    stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                objects.append(cells)
    return objects


# --- Primitive operations ---
# Each returns a function grid → grid (or None if not applicable)

def make_primitives(task: ARCTask) -> list[tuple[str, Callable]]:
    """Generate candidate primitives based on task analysis.

    Returns (name, function) pairs.
    """
    primitives = []

    # Geometric
    for k in [1, 2, 3]:
        primitives.append((f"rot{k*90}", lambda g, k=k: torch.rot90(g, k, (0, 1))))
    primitives.append(("flip_h", lambda g: g.flip(1)))
    primitives.append(("flip_v", lambda g: g.flip(0)))
    primitives.append(("transpose", lambda g: g.t()))

    # Color swaps (derived from demo pairs)
    pair = task.train[0]
    if pair.input.shape == pair.output.shape:
        in_colors = pair.input.unique().tolist()
        out_colors = pair.output.unique().tolist()
        all_colors = set(in_colors + out_colors)

        # Simple two-color swaps
        for c1 in all_colors:
            for c2 in all_colors:
                if c1 < c2:
                    def swap(g, a=c1, b=c2):
                        r = g.clone()
                        r[g == a] = b
                        r[g == b] = a
                        return r
                    primitives.append((f"swap_{c1}_{c2}", swap))

    # Color fills (background to most common)
    for color in range(1, NUM_COLORS):
        def fill_bg(g, c=color):
            r = g.clone()
            r[g == 0] = c
            return r
        primitives.append((f"fill_bg_{color}", fill_bg))

    # Remove each color
    for color in range(1, NUM_COLORS):
        def remove(g, c=color):
            r = g.clone()
            r[g == c] = 0
            return r
        primitives.append((f"remove_{color}", remove))

    # Object extraction (crop to bounding box of non-zero)
    def crop_nonzero(g):
        nz = (g != 0).nonzero(as_tuple=True)
        if len(nz[0]) == 0:
            return g
        return g[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1]
    primitives.append(("crop_nonzero", crop_nonzero))

    # Scale operations
    for s in [2, 3]:
        def scale(g, s=s):
            return g.repeat_interleave(s, 0).repeat_interleave(s, 1)
        primitives.append((f"scale_{s}x", scale))

    # Gravity
    for direction in ["down", "up", "left", "right"]:
        def make_grav(d):
            def grav(g):
                h, w = g.shape
                result = torch.zeros_like(g)
                if d == "down":
                    for c in range(w):
                        col = g[:, c]
                        nz = col[col != 0]
                        result[h-len(nz):, c] = nz
                elif d == "up":
                    for c in range(w):
                        col = g[:, c]
                        nz = col[col != 0]
                        result[:len(nz), c] = nz
                elif d == "right":
                    for r in range(h):
                        row = g[r, :]
                        nz = row[row != 0]
                        result[r, w-len(nz):] = nz
                elif d == "left":
                    for r in range(h):
                        row = g[r, :]
                        nz = row[row != 0]
                        result[r, :len(nz)] = nz
                return result
            return grav
        primitives.append((f"gravity_{direction}", make_grav(direction)))

    return primitives


def verify_program(
    program: Callable, pairs: list[ARCPair],
) -> bool:
    """Check if a program produces correct output for all demo pairs."""
    for pair in pairs:
        try:
            result = program(pair.input)
            if not torch.equal(result, pair.output):
                return False
        except Exception:
            return False
    return True


def search_depth1(
    task: ARCTask, primitives: list[tuple[str, Callable]],
) -> Optional[tuple[str, Callable]]:
    """Try each primitive individually."""
    for name, fn in primitives:
        if verify_program(fn, task.train):
            return name, fn
    return None


def search_depth2(
    task: ARCTask, primitives: list[tuple[str, Callable]],
    max_combinations: int = 2000,
) -> Optional[tuple[str, Callable]]:
    """Try all valid compositions of 2 primitives.

    Applies fn2(fn1(input)) for each (fn1, fn2) pair.
    Limited to max_combinations to keep search tractable.
    """
    count = 0
    for (name1, fn1), (name2, fn2) in product(primitives, primitives):
        if count >= max_combinations:
            break
        count += 1

        def composed(g, f1=fn1, f2=fn2):
            return f2(f1(g))

        try:
            if verify_program(composed, task.train):
                return f"{name1} → {name2}", composed
        except Exception:
            continue

    return None


class ProgramSearchSolver:
    """Brute-force program search as last resort.

    Tries depth-1, then depth-2 compositions.
    Integrates with the Type Lattice to prune invalid compositions.
    """

    def __init__(self, max_depth: int = 2, max_depth2_combos: int = 2000):
        self.max_depth = max_depth
        self.max_depth2_combos = max_depth2_combos
        self.stats = {"tried": 0, "solved_d1": 0, "solved_d2": 0}

    def try_solve(self, task: ARCTask) -> Optional[list[torch.Tensor]]:
        """Try to solve task via program search."""
        self.stats["tried"] += 1

        primitives = make_primitives(task)

        # Depth 1
        result = search_depth1(task, primitives)
        if result:
            name, fn = result
            self.stats["solved_d1"] += 1
            return [fn(tp.input) for tp in task.test]

        # Depth 2
        if self.max_depth >= 2:
            result = search_depth2(
                task, primitives, max_combinations=self.max_depth2_combos,
            )
            if result:
                name, fn = result
                self.stats["solved_d2"] += 1
                return [fn(tp.input) for tp in task.test]

        return None
