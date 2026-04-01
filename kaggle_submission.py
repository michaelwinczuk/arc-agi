"""Kaggle Submission Script for ARC-AGI-2.

This runs as a Kaggle notebook. It:
1. Loads the pre-trained Lattice solver
2. Runs TTT (test-time training) per puzzle
3. Generates pass@2 predictions
4. Writes submission.json

Kaggle constraints:
- 4x L4 GPUs, 12 hours total
- 240 hidden tasks
- pass@2 scoring (2 attempts per task)

Budget at 55ms/puzzle primary + 2s TTT:
  240 * 2s = 8 minutes (well within 12 hours)
"""

import json
import time
import os
from pathlib import Path

import torch


def run_submission():
    t_start = time.time()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        n_gpus = torch.cuda.device_count()
        print(f"GPUs available: {n_gpus}")

    # --- Paths ---
    # Kaggle competition data path
    COMPETITION_DIR = Path("/kaggle/input/arc-prize-2025")
    # Our model weights (uploaded as Kaggle dataset)
    MODEL_DIR = Path("/kaggle/input/lattice-arc-weights")

    # Fallback for local testing
    if not COMPETITION_DIR.exists():
        COMPETITION_DIR = Path("data/ARC-AGI-2/data")
        MODEL_DIR = Path("checkpoints")

    # Find test tasks
    test_dir = COMPETITION_DIR / "test"
    if not test_dir.exists():
        # Local eval mode
        test_dir = COMPETITION_DIR / "evaluation"

    print(f"Test dir: {test_dir}")
    print(f"Model dir: {MODEL_DIR}")

    # --- Load Model ---
    from lattice.data.arc_dataset import load_dataset
    from lattice.evaluate import load_solver
    from lattice.models.ttt import TestTimeTrainer, PassAtTwoSolver

    checkpoint = MODEL_DIR / "best.pt"
    solver = load_solver(
        checkpoint, device,
        num_slots=16, d_slot=64, d_model=64, d_vsa=10000,
    )
    print(f"Model loaded from {checkpoint}")

    # --- Load Tasks ---
    tasks = load_dataset(test_dir)
    print(f"Loaded {len(tasks)} tasks")

    # --- TTT + Solve ---
    ttt = TestTimeTrainer(
        base_solver=solver,
        device=device,
        slot_refine_steps=10,
        slot_refine_lr=1e-3,
        full_adapt_steps=30,
        full_adapt_lr=1e-4,
        verification_threshold=0.5,
    )
    pass2 = PassAtTwoSolver(ttt, device)

    submission = {}
    solve_times = []

    for i, task in enumerate(tasks):
        t0 = time.perf_counter()

        try:
            attempts = pass2.solve_pass_at_2(task)

            task_submission = []
            for test_idx, test_pair in enumerate(task.test):
                if test_idx < len(attempts):
                    a1, a2 = attempts[test_idx]
                    task_submission.append({
                        "attempt_1": a1.cpu().tolist(),
                        "attempt_2": a2.cpu().tolist(),
                    })
                else:
                    # Fallback: empty grid
                    h, w = test_pair.input.shape
                    empty = [[0] * w for _ in range(h)]
                    task_submission.append({
                        "attempt_1": empty,
                        "attempt_2": empty,
                    })

            submission[task.task_id] = task_submission

        except Exception as e:
            print(f"  ERROR {task.task_id}: {e}")
            for test_pair in task.test:
                h, w = test_pair.input.shape
                empty = [[0] * w for _ in range(h)]
                submission[task.task_id] = [{"attempt_1": empty, "attempt_2": empty}]

        elapsed = time.perf_counter() - t0
        solve_times.append(elapsed)

        if (i + 1) % 20 == 0:
            avg = sum(solve_times) / len(solve_times)
            total_elapsed = time.time() - t_start
            eta = avg * (len(tasks) - i - 1)
            print(f"  [{i+1}/{len(tasks)}] avg {avg:.2f}s/task, "
                  f"elapsed {total_elapsed:.0f}s, ETA {eta:.0f}s, "
                  f"library: {solver.library.size} ops")

    # --- Write Submission ---
    output_path = Path("/kaggle/working/submission.json")
    if not output_path.parent.exists():
        output_path = Path("submission.json")

    with open(output_path, "w") as f:
        json.dump(submission, f)

    total_time = time.time() - t_start
    avg_time = sum(solve_times) / max(len(solve_times), 1)
    print(f"\nDone! {len(tasks)} tasks in {total_time:.0f}s "
          f"({avg_time:.2f}s/task avg)")
    print(f"Submission saved to {output_path}")
    print(f"Library: {solver.library.size} cached ops")


if __name__ == "__main__":
    run_submission()
