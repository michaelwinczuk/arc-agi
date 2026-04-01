"""Test-Time Training (TTT) for per-puzzle adaptation.

This is our key speed advantage over NVARC:
- NVARC: LoRA fine-tuning per puzzle (~minutes per puzzle)
- Ours: VSA delta consistency + slot refinement (~seconds per puzzle)

Three TTT strategies, applied in order of increasing cost:

1. VSA Delta Consensus (free, ~0.1ms)
   - Already built into the pipeline
   - Majority vote across demo pair deltas
   - If consistent (Hamming < threshold): done

2. Slot Refinement (cheap, ~50ms)
   - Fine-tune slot attention on the demo pairs for this specific task
   - Objective: minimize reconstruction loss on demo inputs AND outputs
   - 5-10 gradient steps, only update slot initialization

3. Full Adaptation (expensive, ~2s)
   - Fine-tune entire model on demo pairs
   - Objective: minimize output prediction loss
   - 20-50 gradient steps with small LR
   - Only used when strategies 1-2 fail verification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from ..data.arc_dataset import (
    ARCTask, ARCPair, grid_to_onehot, pad_grid, NUM_COLORS,
)


class TestTimeTrainer:
    """Per-puzzle test-time adaptation.

    Takes a pre-trained LatticeSolver and adapts it to a specific task.
    Returns adapted solver (does not modify the original).
    """

    def __init__(
        self,
        base_solver: nn.Module,
        device: torch.device,
        slot_refine_steps: int = 10,
        slot_refine_lr: float = 1e-3,
        full_adapt_steps: int = 30,
        full_adapt_lr: float = 1e-4,
        verification_threshold: float = 0.5,
    ):
        self.base_solver = base_solver
        self.device = device
        self.slot_refine_steps = slot_refine_steps
        self.slot_refine_lr = slot_refine_lr
        self.full_adapt_steps = full_adapt_steps
        self.full_adapt_lr = full_adapt_lr
        self.verification_threshold = verification_threshold

    def _encode_grid(self, grid: torch.Tensor, model: nn.Module):
        """Encode a grid using the model's slot attention."""
        h, w = grid.shape
        padded = pad_grid(grid)
        onehot = grid_to_onehot(padded).unsqueeze(0).to(self.device)
        mask = (padded >= 0).unsqueeze(0).to(self.device)
        slots, attn = model.slot_attention(onehot, mask)
        return slots, attn, h, w

    def _verify_on_demos(
        self, model: nn.Module, task: ARCTask,
    ) -> float:
        """Check accuracy on leave-one-out demo pairs.

        For each demo pair, use the others as training and predict this one.
        Returns fraction correct.
        """
        if len(task.train) < 2:
            return 0.0

        model.eval()
        correct = 0

        with torch.no_grad():
            for i, target_pair in enumerate(task.train):
                # Use all other pairs as demos
                other_pairs = [p for j, p in enumerate(task.train) if j != i]
                test_task = ARCTask(
                    task_id=task.task_id,
                    train=other_pairs,
                    test=[target_pair],
                )

                preds = model.solve_task(test_task, self.device)
                if preds and torch.equal(preds[0].cpu(), target_pair.output):
                    correct += 1

        return correct / len(task.train)

    def _reconstruction_loss(
        self, model: nn.Module, pair: ARCPair,
    ) -> torch.Tensor:
        """Compute reconstruction loss for a single pair."""
        # Encode and reconstruct input
        in_slots, in_attn, in_h, in_w = self._encode_grid(pair.input, model)
        in_logits = model.decoder(in_slots, in_attn, in_h, in_w)
        in_target = pair.input[:in_h, :in_w].unsqueeze(0).to(self.device)
        loss_in = F.cross_entropy(in_logits, in_target)

        # Encode and reconstruct output
        out_slots, out_attn, out_h, out_w = self._encode_grid(pair.output, model)
        out_logits = model.decoder(out_slots, out_attn, out_h, out_w)
        out_target = pair.output[:out_h, :out_w].unsqueeze(0).to(self.device)
        loss_out = F.cross_entropy(out_logits, out_target)

        return loss_in + loss_out

    def _output_prediction_loss(
        self, model: nn.Module, task: ARCTask,
    ) -> torch.Tensor:
        """Compute output prediction loss across all demo pairs.

        For each pair: encode input → decode to output size → compare with GT.
        """
        losses = []
        for pair in task.train:
            in_slots, in_attn, in_h, in_w = self._encode_grid(pair.input, model)
            out_h, out_w = pair.output.shape
            logits = model.decoder(in_slots, in_attn, out_h, out_w)
            target = pair.output[:out_h, :out_w].unsqueeze(0).to(self.device)
            losses.append(F.cross_entropy(logits, target))
        return torch.stack(losses).mean()

    def slot_refinement(
        self, model: nn.Module, task: ARCTask,
    ) -> nn.Module:
        """Strategy 2: Fine-tune slot initialization on this task's demos.

        Only updates slot_mu and slot_logvar — everything else frozen.
        Very fast: 10 steps × ~5ms = ~50ms.
        """
        model.train()

        # Only optimize slot initialization parameters
        slot_params = [
            model.slot_attention.slot_attn.slot_mu,
            model.slot_attention.slot_attn.slot_logvar,
        ]
        optimizer = torch.optim.Adam(slot_params, lr=self.slot_refine_lr)

        for step in range(self.slot_refine_steps):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=self.device)
            for pair in task.train:
                loss = loss + self._reconstruction_loss(model, pair)
            loss = loss / len(task.train)
            loss.backward()
            optimizer.step()

        model.eval()
        return model

    def full_adaptation(
        self, model: nn.Module, task: ARCTask,
    ) -> nn.Module:
        """Strategy 3: Fine-tune entire model on this task's demos.

        Updates all parameters. More expensive but handles harder puzzles.
        30 steps × ~15ms = ~450ms.
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.full_adapt_lr)

        for step in range(self.full_adapt_steps):
            optimizer.zero_grad()

            # Combined loss: reconstruction + output prediction
            recon_loss = torch.tensor(0.0, device=self.device)
            for pair in task.train:
                recon_loss = recon_loss + self._reconstruction_loss(model, pair)
            recon_loss = recon_loss / len(task.train)

            output_loss = self._output_prediction_loss(model, task)

            total = recon_loss + output_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        return model

    def adapt_and_solve(
        self, task: ARCTask,
    ) -> list[torch.Tensor]:
        """Full TTT pipeline for a single task.

        1. Try base model (no adaptation)
        2. If verification fails → slot refinement
        3. If still fails → full adaptation
        4. Return best predictions

        Returns list of predicted output grids.
        """
        # Strategy 1: Base model with VSA consensus
        base_acc = self._verify_on_demos(self.base_solver, task)

        if base_acc >= self.verification_threshold:
            # Base model works, no adaptation needed
            return self.base_solver.solve_task(task, self.device)

        # Strategy 2: Slot refinement
        refined = deepcopy(self.base_solver)
        refined = self.slot_refinement(refined, task)
        refined_acc = self._verify_on_demos(refined, task)

        if refined_acc >= self.verification_threshold:
            return refined.solve_task(task, self.device)

        # Strategy 3: Full adaptation
        adapted = deepcopy(self.base_solver)
        adapted = self.full_adaptation(adapted, task)
        adapted_acc = self._verify_on_demos(adapted, task)

        # Return whichever did best
        best_model = max(
            [(self.base_solver, base_acc),
             (refined, refined_acc),
             (adapted, adapted_acc)],
            key=lambda x: x[1],
        )[0]

        return best_model.solve_task(task, self.device)


class PassAtTwoSolver:
    """Generate two attempts per task for pass@2 scoring.

    ARC-AGI-2 allows 2 guesses per puzzle. We use:
    - Attempt 1: Best prediction from TTT pipeline
    - Attempt 2: Prediction with geometric augmentation + inverse

    This doubles our effective accuracy for puzzles where we're close.
    """

    def __init__(
        self,
        ttt: TestTimeTrainer,
        device: torch.device,
    ):
        self.ttt = ttt
        self.device = device

    def solve_pass_at_2(
        self, task: ARCTask,
    ) -> list[list[torch.Tensor]]:
        """Generate 2 attempts per test input.

        Returns list of [attempt1, attempt2] per test input.
        """
        # Attempt 1: Standard TTT
        attempt1 = self.ttt.adapt_and_solve(task)

        # Attempt 2: Solve with augmented task (rotate 90°),
        # then inverse-rotate the prediction
        from ..utils.augmentation import rotate_grid, augment_task

        aug_task = augment_task(
            task,
            lambda g: rotate_grid(g, 1),
            suffix="_rot90",
        )
        aug_preds = self.ttt.adapt_and_solve(aug_task)

        # Inverse rotation on predictions
        attempt2 = [rotate_grid(p, -1) for p in aug_preds]

        # If both attempts are identical, try a different augmentation
        results = []
        for a1, a2 in zip(attempt1, attempt2):
            if torch.equal(a1, a2):
                # Try horizontal flip instead
                flip_task = augment_task(
                    task,
                    lambda g: g.flip(1),
                    suffix="_fliph",
                )
                flip_preds = self.ttt.adapt_and_solve(flip_task)
                a2 = flip_preds[0].flip(1) if flip_preds else a2
            results.append([a1, a2])

        return results
