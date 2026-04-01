"""Evaluation script: run trained model on ARC-AGI-2 eval tasks.

Loads a checkpoint, runs inference on all 120 evaluation tasks,
reports per-task and aggregate accuracy.

Usage:
    python -m lattice.evaluate --data_dir data/ARC-AGI-2/data/evaluation \
        --checkpoint checkpoints/best.pt --device cuda
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import torch

from .data.arc_dataset import load_dataset, ARCTask
from .models.pipeline import LatticeSolver, RefinementLoop
from .utils.visualization import print_grid


def load_solver(
    checkpoint_path: Path,
    device: torch.device,
    num_slots: int = 16,
    d_slot: int = 64,
    d_model: int = 64,
    d_vsa: int = 10000,
) -> LatticeSolver:
    """Load a trained solver from checkpoint."""
    solver = LatticeSolver(
        num_slots=num_slots,
        d_slot=d_slot,
        d_model=d_model,
        d_vsa=d_vsa,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both full trainer checkpoints and solver-only checkpoints
    state_dict = ckpt.get("model_state_dict", ckpt)

    # If trained with LatticeTrainer wrapper, strip prefix
    solver_state = {}
    for k, v in state_dict.items():
        # LatticeTrainer has same module names as LatticeSolver
        # (slot_attention, decoder, delta_extractor, etc.)
        solver_state[k] = v

    # Try loading directly first, then with flexible matching
    try:
        solver.load_state_dict(solver_state, strict=True)
    except RuntimeError:
        # Trainer and Solver share architecture, load what matches
        own_state = solver.state_dict()
        loaded = 0
        for name, param in solver_state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
        print(f"Loaded {loaded}/{len(own_state)} parameters (flexible mode)")

    solver.eval()
    return solver


def evaluate_task(
    solver: LatticeSolver,
    task: ARCTask,
    device: torch.device,
    use_refinement: bool = True,
    verbose: bool = False,
) -> dict:
    """Evaluate solver on a single task.

    Returns dict with:
        correct: bool (all test outputs match)
        per_test: list of bools
        time_ms: float
    """
    t0 = time.perf_counter()

    predictions = solver.solve_task(task, device)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    per_test = []
    for j, (pred, test_pair) in enumerate(zip(predictions, task.test)):
        if test_pair.output is None:
            per_test.append(None)
            continue

        match = torch.equal(pred.cpu(), test_pair.output)
        per_test.append(match)

        if verbose and not match:
            print(f"  Test {j+1}: WRONG")
            print(f"    Expected ({test_pair.output.shape[0]}x{test_pair.output.shape[1]}):")
            print_grid(test_pair.output)
            print(f"    Got ({pred.shape[0]}x{pred.shape[1]}):")
            print_grid(pred.cpu())

    scorable = [p for p in per_test if p is not None]
    all_correct = all(scorable) if scorable else False

    return {
        "correct": all_correct,
        "per_test": per_test,
        "time_ms": elapsed_ms,
    }


def evaluate_dataset(
    solver: LatticeSolver,
    tasks: list[ARCTask],
    device: torch.device,
    verbose: bool = False,
) -> dict:
    """Evaluate solver on full dataset.

    Returns comprehensive results dict.
    """
    results = {
        "tasks": {},
        "correct": 0,
        "total": 0,
        "total_time_ms": 0,
        "errors": 0,
    }

    for i, task in enumerate(tasks):
        try:
            task_result = evaluate_task(
                solver, task, device, verbose=verbose,
            )
            results["tasks"][task.task_id] = task_result
            results["total"] += 1
            results["total_time_ms"] += task_result["time_ms"]

            if task_result["correct"]:
                results["correct"] += 1

            if (i + 1) % 10 == 0 or verbose:
                acc = results["correct"] / results["total"] * 100
                avg_ms = results["total_time_ms"] / results["total"]
                print(f"  [{i+1}/{len(tasks)}] {results['correct']}/{results['total']} "
                      f"({acc:.1f}%) avg {avg_ms:.0f}ms/task "
                      f"library: {solver.library.size} ops")

        except Exception as e:
            results["errors"] += 1
            results["total"] += 1
            if verbose:
                print(f"  [{i+1}] {task.task_id}: ERROR - {e}")

    # Summary
    acc = results["correct"] / max(results["total"], 1) * 100
    avg_ms = results["total_time_ms"] / max(results["total"], 1)
    results["accuracy"] = acc
    results["avg_ms"] = avg_ms

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI-2 Lattice Solver")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--d_slot", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_vsa", type=int, default=10000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    solver = load_solver(
        Path(args.checkpoint), device,
        num_slots=args.num_slots, d_slot=args.d_slot,
        d_model=args.d_model, d_vsa=args.d_vsa,
    )
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load tasks
    tasks = load_dataset(Path(args.data_dir))
    print(f"Loaded {len(tasks)} tasks")

    # Evaluate
    results = evaluate_dataset(solver, tasks, device, verbose=args.verbose)

    print(f"\n{'='*50}")
    print(f"Results: {results['correct']}/{results['total']} "
          f"({results['accuracy']:.1f}%)")
    print(f"Average: {results['avg_ms']:.0f}ms/task")
    print(f"Errors: {results['errors']}")
    print(f"Library: {solver.library.size} ops cached")

    if args.output:
        # Serialize (strip non-JSON-serializable fields)
        out = {
            "correct": results["correct"],
            "total": results["total"],
            "accuracy": results["accuracy"],
            "avg_ms": results["avg_ms"],
            "errors": results["errors"],
            "per_task": {
                tid: {"correct": r["correct"], "time_ms": r["time_ms"]}
                for tid, r in results["tasks"].items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
