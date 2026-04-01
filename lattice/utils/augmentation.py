"""Grid augmentation utilities for ARC tasks.

Geometric augmentations that preserve task semantics:
- 4 rotations (0, 90, 180, 270)
- 2 reflections (horizontal, vertical)
- Color permutations (keeping black=0 fixed)

Used for:
1. Training data augmentation
2. Test-time augmentation for re-scoring (NVARC-style)
3. Verification: if solution is correct, it should survive augmentation
"""

import torch
import itertools
from typing import Iterator

from ..data.arc_dataset import ARCPair, ARCTask, NUM_COLORS


def rotate_grid(grid: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate grid by k*90 degrees counterclockwise."""
    return torch.rot90(grid, k, dims=(0, 1))


def flip_h(grid: torch.Tensor) -> torch.Tensor:
    """Flip grid horizontally."""
    return grid.flip(1)


def flip_v(grid: torch.Tensor) -> torch.Tensor:
    """Flip grid vertically."""
    return grid.flip(0)


def permute_colors(grid: torch.Tensor, perm: list[int]) -> torch.Tensor:
    """Apply color permutation. perm[i] = new color for original color i.
    Color 0 (black/background) is always fixed.
    """
    result = grid.clone()
    for old_color, new_color in enumerate(perm):
        if old_color != new_color:
            result[grid == old_color] = new_color
    return result


def geometric_augmentations(
    grid: torch.Tensor,
) -> list[tuple[torch.Tensor, str]]:
    """Generate all 8 geometric augmentations (D4 group).

    Returns list of (augmented_grid, augmentation_name).
    """
    augs = []
    for k in range(4):
        rotated = rotate_grid(grid, k)
        augs.append((rotated, f"rot{k*90}"))
        augs.append((flip_h(rotated), f"rot{k*90}_fliph"))
    return augs


def augment_pair(pair: ARCPair, aug_fn) -> ARCPair:
    """Apply augmentation function to both input and output of a pair."""
    return ARCPair(
        input=aug_fn(pair.input),
        output=aug_fn(pair.output) if pair.output is not None else None,
    )


def augment_task(task: ARCTask, aug_fn, suffix: str = "") -> ARCTask:
    """Apply augmentation to all pairs in a task."""
    return ARCTask(
        task_id=f"{task.task_id}{suffix}",
        train=[augment_pair(p, aug_fn) for p in task.train],
        test=[augment_pair(p, aug_fn) for p in task.test],
    )


def generate_color_permutations(
    n_perms: int = 8, seed: int = 42
) -> list[list[int]]:
    """Generate random color permutations keeping 0 fixed.

    Returns list of permutations, each is a list of length NUM_COLORS.
    """
    rng = torch.Generator().manual_seed(seed)
    perms = []
    for _ in range(n_perms):
        # Permute colors 1-9, keep 0 fixed
        p = list(range(NUM_COLORS))
        non_zero = torch.randperm(NUM_COLORS - 1, generator=rng) + 1
        for i, v in enumerate(non_zero.tolist()):
            p[i + 1] = v
        perms.append(p)
    return perms
