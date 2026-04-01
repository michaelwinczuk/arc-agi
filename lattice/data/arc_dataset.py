"""ARC-AGI-2 dataset loader.

Loads tasks from the official ARC-AGI-2 JSON format.
Each task has 'train' (demo pairs) and 'test' (input only) sections.
Grids are 2D arrays of ints 0-9 (10 colors), up to 30x30.
"""

import json
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F


MAX_GRID_SIZE = 30
NUM_COLORS = 10


@dataclass
class ARCPair:
    input: torch.Tensor   # (H, W) int64, values 0-9
    output: torch.Tensor  # (H, W) int64, values 0-9


@dataclass
class ARCTask:
    task_id: str
    train: list[ARCPair]  # demo pairs (2-10 typically)
    test: list[ARCPair]   # test pairs (usually 1)


def grid_to_tensor(grid: list[list[int]]) -> torch.Tensor:
    return torch.tensor(grid, dtype=torch.long)


def pad_grid(grid: torch.Tensor, size: int = MAX_GRID_SIZE) -> torch.Tensor:
    """Pad grid to fixed size with -1 (mask value). Returns (size, size)."""
    h, w = grid.shape
    padded = torch.full((size, size), -1, dtype=torch.long)
    padded[:h, :w] = grid
    return padded


def grid_to_onehot(grid: torch.Tensor) -> torch.Tensor:
    """Convert (H, W) int grid to (NUM_COLORS, H, W) float one-hot encoding.
    Cells with value -1 (padding) map to all-zeros."""
    h, w = grid.shape
    mask = grid >= 0
    safe_grid = grid.clamp(min=0)
    onehot = F.one_hot(safe_grid, NUM_COLORS).permute(2, 0, 1).float()
    onehot = onehot * mask.unsqueeze(0).float()
    return onehot


def load_task(path: Path) -> ARCTask:
    """Load a single ARC task from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    task_id = path.stem

    train = []
    for pair in data["train"]:
        train.append(ARCPair(
            input=grid_to_tensor(pair["input"]),
            output=grid_to_tensor(pair["output"]),
        ))

    test = []
    for pair in data["test"]:
        inp = grid_to_tensor(pair["input"])
        out = grid_to_tensor(pair["output"]) if "output" in pair else None
        test.append(ARCPair(input=inp, output=out))

    return ARCTask(task_id=task_id, train=train, test=test)


def load_dataset(directory: Path) -> list[ARCTask]:
    """Load all ARC tasks from a directory of JSON files."""
    tasks = []
    for path in sorted(directory.glob("*.json")):
        tasks.append(load_task(path))
    return tasks
