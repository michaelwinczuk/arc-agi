"""Main entry point for solving ARC-AGI-2 tasks.

Usage:
    python -m lattice.solve --data_dir /path/to/arc-agi-2/tasks --output submissions.json

For Kaggle submission:
    This module is imported by the submission notebook.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from .data.arc_dataset import load_dataset, ARCTask
from .models.pipeline import LatticeSolver, RefinementLoop


def solve_all(
    tasks: list[ARCTask],
    device: torch.device,
    num_attempts: int = 2,
    verbose: bool = True,
) -> dict[str, list[list[list[int]]]]:
    """Solve all tasks and return submission dict.

    ARC-AGI-2 allows pass@2: 2 attempts per task.

    Args:
        tasks: list of ARCTask
        device: compute device
        num_attempts: number of attempts per task (max 2 for competition)
        verbose: print progress
    Returns:
        Dict mapping task_id -> list of attempts (each attempt is a grid)
    """
    solver = LatticeSolver().to(device)
    solver.eval()

    refinement = RefinementLoop(solver, max_iters=3)
    submissions = {}

    total_time = 0

    for i, task in enumerate(tasks):
        t0 = time.perf_counter()

        try:
            predictions = solver.solve_task(task, device)

            # For pass@2: generate a second attempt with slight variation
            attempts = []
            for pred in predictions:
                grid_list = pred.cpu().tolist()
                attempts.append(grid_list)

            submissions[task.task_id] = attempts

        except Exception as e:
            if verbose:
                print(f"  ERROR on {task.task_id}: {e}")
            # Submit empty grid as fallback
            for test_pair in task.test:
                h, w = test_pair.input.shape
                submissions[task.task_id] = [[[0] * w for _ in range(h)]]

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        if verbose and (i + 1) % 10 == 0:
            avg = total_time / (i + 1)
            print(f"  [{i+1}/{len(tasks)}] avg {avg*1000:.0f}ms/task, "
                  f"library: {solver.library.size} ops")

    if verbose:
        print(f"\nDone. {len(tasks)} tasks in {total_time:.1f}s "
              f"({total_time/max(len(tasks),1)*1000:.0f}ms/task avg)")
        print(f"Library: {solver.library.stats()}")

    return submissions


def evaluate(
    tasks: list[ARCTask],
    submissions: dict[str, list[list[list[int]]]],
) -> dict:
    """Evaluate predictions against ground truth.

    Only works for tasks that have test outputs (local eval).
    """
    correct = 0
    total = 0

    for task in tasks:
        for j, test_pair in enumerate(task.test):
            if test_pair.output is None:
                continue
            total += 1
            gt = test_pair.output.tolist()

            task_attempts = submissions.get(task.task_id, [])
            if j < len(task_attempts):
                pred = task_attempts[j]
                if pred == gt:
                    correct += 1

    accuracy = correct / max(total, 1)
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "percentage": f"{accuracy * 100:.1f}%",
    }


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Lattice Solver")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing ARC task JSON files")
    parser.add_argument("--output", type=str, default="submission.json",
                        help="Output submission file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation on tasks with known outputs")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    tasks = load_dataset(data_dir)
    print(f"Loaded {len(tasks)} tasks from {data_dir}")

    submissions = solve_all(tasks, device)

    # Save submission
    with open(args.output, "w") as f:
        json.dump(submissions, f)
    print(f"Saved to {args.output}")

    if args.eval:
        results = evaluate(tasks, submissions)
        print(f"\nEvaluation: {results['correct']}/{results['total']} "
              f"({results['percentage']})")


if __name__ == "__main__":
    main()
