"""Grid Decoder: reconstruct ARC output grid from predicted VSA slots.

Takes predicted output slot vectors and attention maps,
produces a discrete 10-color grid.

Two decoding paths:
1. Attention-based: use slot-to-position attention to assign colors
2. VSA nearest-neighbor: find closest known grid encoding in library
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.arc_dataset import NUM_COLORS, MAX_GRID_SIZE


class SlotDecoder(nn.Module):
    """Decode slots back to a grid via learned spatial broadcast.

    Each slot broadcasts its features to all positions,
    weighted by attention. A final conv predicts per-pixel colors.
    """

    def __init__(self, d_slot: int = 64, d_model: int = 64):
        super().__init__()
        self.d_slot = d_slot

        # Project slots to spatial features
        self.slot_project = nn.Linear(d_slot, d_model)

        # Decode mixed features to color logits
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_COLORS),
        )

    def forward(
        self,
        slots: torch.Tensor,
        attn: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Args:
            slots: (B, K, D_slot) predicted output slots
            attn: (B, K, N) slot-to-position attention weights
            grid_h, grid_w: output grid dimensions
        Returns:
            logits: (B, NUM_COLORS, H, W) color logits per position
        """
        B, K, D = slots.shape
        N = grid_h * grid_w

        # Project slots
        slot_features = self.slot_project(slots)  # (B, K, D_model)

        # Weighted combination: each position gets mixture of slot features
        # attn: (B, K, N) -> transpose to (B, N, K)
        attn_t = attn[:, :, :N].transpose(1, 2)  # (B, N, K)

        # Mix: (B, N, K) @ (B, K, D_model) -> (B, N, D_model)
        mixed = torch.bmm(attn_t, slot_features)

        # Decode to color logits
        logits = self.decoder(mixed)  # (B, N, NUM_COLORS)
        logits = logits.transpose(1, 2)  # (B, NUM_COLORS, N)
        logits = logits.view(B, NUM_COLORS, grid_h, grid_w)

        return logits

    def decode_grid(
        self,
        slots: torch.Tensor,
        attn: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """Decode to discrete grid (argmax).

        Returns:
            grid: (B, H, W) int64 values 0-9
        """
        logits = self.forward(slots, attn, grid_h, grid_w)
        return logits.argmax(dim=1)  # (B, H, W)
