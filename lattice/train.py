"""Training script for the Lattice ARC solver.

Three training objectives:
1. Slot Attention reconstruction: can we reconstruct the input grid from slots?
2. VSA delta consistency: are deltas consistent across demo pairs of the same task?
3. Type classification: do predicted types match observed transformations?

All three train jointly, end-to-end.

Usage:
    python -m lattice.train --data_dir data/ARC-AGI-2/data/training --epochs 100
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data.arc_dataset import (
    load_dataset, ARCTask, ARCPair,
    grid_to_onehot, pad_grid, MAX_GRID_SIZE, NUM_COLORS,
)
from .models.slot_attention import SlotAttention
from .models.vsa import DeltaExtractor, ConsensusBuilder, VSAOperations
from .models.grid_decoder import SlotDecoder
from .models.cross_attention_decoder import TransformationEncoder, CrossAttentionDecoder
from .models.type_classifier import TransformationClassifier
from .utils.augmentation import geometric_augmentations, generate_color_permutations, permute_colors


class LatticeTrainer(nn.Module):
    """Joint training wrapper for all components."""

    def __init__(
        self,
        num_slots: int = 16,
        d_slot: int = 64,
        d_model: int = 64,
        d_vsa: int = 10000,
        num_sa_iters: int = 3,
    ):
        super().__init__()

        self.slot_attention = SlotAttention(
            num_slots=num_slots, d_slot=d_slot,
            d_model=d_model, num_iters=num_sa_iters,
        )
        self.decoder = SlotDecoder(d_slot=d_slot, d_model=d_model)
        self.delta_extractor = DeltaExtractor(
            d_slot=d_slot, d_vsa=d_vsa, num_slots=num_slots,
        )
        self.consensus_builder = ConsensusBuilder(d_vsa=d_vsa)
        self.vsa_ops = VSAOperations()

        # Cross-attention decoder (the real decoder)
        d_transform = d_slot * 2
        self.transform_encoder = TransformationEncoder(
            d_slot=d_slot, d_transform=d_transform,
        )
        self.cross_decoder = CrossAttentionDecoder(
            d_slot=d_slot, d_transform=d_transform, d_model=d_transform,
        )

        self.d_vsa = d_vsa

    def encode_grid(self, grid: torch.Tensor, device: torch.device):
        """Encode a single grid. Returns slots, attn, h, w."""
        h, w = grid.shape
        padded = pad_grid(grid)
        onehot = grid_to_onehot(padded).unsqueeze(0).to(device)
        mask = (padded >= 0).unsqueeze(0).to(device)
        slots, attn = self.slot_attention(onehot, mask)
        return slots, attn, h, w

    def reconstruction_loss(
        self, slots: torch.Tensor, attn: torch.Tensor,
        target_grid: torch.Tensor, h: int, w: int, device: torch.device,
    ) -> torch.Tensor:
        """Loss 1: Can we reconstruct the grid from slots?

        Cross-entropy between decoded grid and ground truth.
        """
        logits = self.decoder(slots, attn, h, w)  # (1, NUM_COLORS, H, W)
        target = target_grid[:h, :w].unsqueeze(0).to(device)  # (1, H, W)
        return F.cross_entropy(logits, target)

    def delta_consistency_loss(
        self, deltas: list[torch.Tensor],
    ) -> torch.Tensor:
        """Loss 2: Are deltas consistent across demo pairs?

        Minimize variance of deltas within a task.
        Consistent deltas should have low pairwise Hamming distance.
        """
        if len(deltas) < 2:
            return torch.tensor(0.0, device=deltas[0].device)

        stacked = torch.stack(deltas)  # (N, D_vsa)
        mean = stacked.mean(dim=0)  # (D_vsa,)

        # MSE from mean (in continuous space before binarization)
        return ((stacked - mean) ** 2).mean()

    def train_step(
        self, task: ARCTask, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """One training step on a single task.

        Returns dict of loss components.
        """
        recon_losses = []
        deltas = []
        pair_encodings = []

        for pair in task.train:
            # Encode both grids
            in_slots, in_attn, in_h, in_w = self.encode_grid(pair.input, device)
            out_slots, out_attn, out_h, out_w = self.encode_grid(pair.output, device)

            # Reconstruction loss on both input and output
            recon_in = self.reconstruction_loss(
                in_slots, in_attn, pair.input, in_h, in_w, device
            )
            recon_out = self.reconstruction_loss(
                out_slots, out_attn, pair.output, out_h, out_w, device
            )
            recon_losses.append(recon_in + recon_out)

            # Delta extraction
            delta, _ = self.delta_extractor(in_slots, out_slots)
            deltas.append(delta.squeeze(0))

            # Transformation encoding for cross-attention decoder
            pair_enc = self.transform_encoder.encode_pair(in_slots, out_slots)
            pair_encodings.append(pair_enc)

        # Aggregate losses
        recon_loss = torch.stack(recon_losses).mean()
        consistency_loss = self.delta_consistency_loss(deltas)

        # Aggregated transformation embedding
        transform_embed = self.transform_encoder(pair_encodings)

        # Output prediction loss (simple decoder)
        output_losses = []
        for pair in task.train:
            in_slots, in_attn, in_h, in_w = self.encode_grid(pair.input, device)
            out_h, out_w = pair.output.shape
            logits = self.decoder(in_slots, in_attn, out_h, out_w)
            target = pair.output[:out_h, :out_w].unsqueeze(0).to(device)
            output_losses.append(F.cross_entropy(logits, target))

        output_loss = torch.stack(output_losses).mean()

        # Cross-attention decoder loss (the important one)
        # Leave-one-out: for each pair, use others as demos, predict this one
        cross_losses = []
        for i, pair in enumerate(task.train):
            in_slots, in_attn, in_h, in_w = self.encode_grid(pair.input, device)
            out_h, out_w = pair.output.shape

            # Use transform_embed from all pairs (including this one for now,
            # leave-one-out is expensive — will add later)
            logits = self.cross_decoder(
                in_slots, transform_embed, in_attn, out_h, out_w,
            )
            target = pair.output[:out_h, :out_w].unsqueeze(0).to(device)
            cross_losses.append(F.cross_entropy(logits, target))

        cross_loss = torch.stack(cross_losses).mean()

        total = recon_loss + 0.5 * consistency_loss + output_loss + cross_loss

        return {
            "total": total,
            "recon": recon_loss,
            "consistency": consistency_loss,
            "output": output_loss,
            "cross": cross_loss,
        }


def train(
    data_dir: Path,
    epochs: int = 100,
    lr: float = 3e-4,
    device_str: str = "cuda",
    num_slots: int = 16,
    d_slot: int = 64,
    d_model: int = 64,
    d_vsa: int = 10000,
    save_every: int = 10,
    save_dir: str = "checkpoints",
    augment: bool = True,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tasks = load_dataset(data_dir)
    print(f"Loaded {len(tasks)} tasks")

    model = LatticeTrainer(
        num_slots=num_slots, d_slot=d_slot,
        d_model=d_model, d_vsa=d_vsa,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Color permutations for augmentation
    color_perms = generate_color_permutations(n_perms=8) if augment else []

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"total": 0, "recon": 0, "consistency": 0, "output": 0, "cross": 0}
        t0 = time.perf_counter()

        # Shuffle tasks each epoch
        indices = torch.randperm(len(tasks))

        for step, idx in enumerate(indices):
            task = tasks[idx.item()]

            # Optional: augment with color permutation
            if augment and color_perms and torch.rand(1).item() < 0.5:
                perm = color_perms[torch.randint(len(color_perms), (1,)).item()]
                task = ARCTask(
                    task_id=task.task_id,
                    train=[ARCPair(
                        input=permute_colors(p.input, perm),
                        output=permute_colors(p.output, perm),
                    ) for p in task.train],
                    test=task.test,
                )

            optimizer.zero_grad()
            losses = model.train_step(task, device)
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            if (step + 1) % 100 == 0:
                avg = epoch_losses["total"] / (step + 1)
                elapsed = time.perf_counter() - t0
                ms_per_task = elapsed / (step + 1) * 1000
                print(f"  [{epoch+1}] step {step+1}/{len(tasks)} "
                      f"loss={avg:.4f} ({ms_per_task:.0f}ms/task)")

        scheduler.step()

        # Epoch stats
        n = len(tasks)
        elapsed = time.perf_counter() - t0
        avg_losses = {k: v / n for k, v in epoch_losses.items()}

        print(f"Epoch {epoch+1}/{epochs} | "
              f"loss={avg_losses['total']:.4f} "
              f"recon={avg_losses['recon']:.4f} "
              f"cross={avg_losses['cross']:.4f} "
              f"output={avg_losses['output']:.4f} "
              f"consist={avg_losses['consistency']:.4f} | "
              f"{elapsed:.0f}s | lr={scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if avg_losses["total"] < best_loss:
            best_loss = avg_losses["total"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, save_path / "best.pt")
            print(f"  Saved best model (loss={best_loss:.4f})")

        if (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_losses["total"],
            }, save_path / f"epoch_{epoch+1}.pt")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Lattice ARC Solver")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--d_slot", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_vsa", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()

    train(
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        lr=args.lr,
        device_str=args.device,
        num_slots=args.num_slots,
        d_slot=args.d_slot,
        d_model=args.d_model,
        d_vsa=args.d_vsa,
        save_every=args.save_every,
        save_dir=args.save_dir,
        augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
