"""Slot Attention for ARC grid object decomposition.

Decomposes an ARC grid into K <= 16 object slots.
Each slot captures a distinct object/region in the grid.

Based on Locatello et al. "Object-Centric Learning with Slot Attention" (2020),
adapted for discrete 10-color grids up to 30x30.

Target: ~50ms per grid on L4 GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.arc_dataset import NUM_COLORS, MAX_GRID_SIZE


class GridEncoder(nn.Module):
    """Encode a one-hot ARC grid into spatial feature maps.

    Input: (B, NUM_COLORS, H, W) one-hot grid
    Output: (B, N, D) where N = H*W spatial positions, D = feature dim
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model

        # Small CNN to extract local features from color channels
        self.encoder = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, d_model, 3, padding=1),
            nn.ReLU(),
        )

        # Learnable 2D positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, d_model, MAX_GRID_SIZE, MAX_GRID_SIZE) * 0.02
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, NUM_COLORS, H, W) one-hot encoded grid
            mask: (B, H, W) bool - True for valid cells
        Returns:
            (B, N_valid, D) spatial features for valid positions only
        """
        B, C, H, W = x.shape

        features = self.encoder(x)  # (B, D, H, W)
        features = features + self.pos_embed[:, :, :H, :W]

        # Flatten spatial dims: (B, D, H, W) -> (B, H*W, D)
        features = features.flatten(2).transpose(1, 2)

        # Mask out padding positions
        flat_mask = mask.flatten(1)  # (B, H*W)
        # Zero out invalid positions (they won't attend)
        features = features * flat_mask.unsqueeze(-1).float()

        return features, flat_mask


class SlotAttentionModule(nn.Module):
    """Iterative slot attention mechanism.

    Slots compete to explain spatial positions via softmax attention.
    Each iteration refines slot representations via GRU update.
    """

    def __init__(
        self,
        num_slots: int = 16,
        d_slot: int = 64,
        d_model: int = 64,
        num_iters: int = 3,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.d_slot = d_slot
        self.num_iters = num_iters
        self.epsilon = epsilon

        # Slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, 1, d_slot) * 0.02)
        self.slot_logvar = nn.Parameter(torch.zeros(1, 1, d_slot))

        # Attention projections
        self.project_q = nn.Linear(d_slot, d_model)
        self.project_k = nn.Linear(d_model, d_model)
        self.project_v = nn.Linear(d_model, d_model)

        # Slot update
        self.gru = nn.GRUCell(d_model, d_slot)
        self.norm_slots = nn.LayerNorm(d_slot)
        self.norm_inputs = nn.LayerNorm(d_model)

        # MLP refinement after GRU
        self.mlp = nn.Sequential(
            nn.Linear(d_slot, d_slot * 2),
            nn.ReLU(),
            nn.Linear(d_slot * 2, d_slot),
        )
        self.norm_mlp = nn.LayerNorm(d_slot)

    def _init_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize slots with learned Gaussian."""
        mu = self.slot_mu.expand(batch_size, self.num_slots, -1)
        logvar = self.slot_logvar.expand(batch_size, self.num_slots, -1)
        std = (logvar * 0.5).exp()
        return mu + std * torch.randn_like(std)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (B, N, D) spatial features
            mask: (B, N) bool mask for valid positions
        Returns:
            slots: (B, K, D_slot) slot representations
            attn: (B, K, N) attention weights (which positions each slot owns)
        """
        B, N, D = inputs.shape
        slots = self._init_slots(B, inputs.device)

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)   # (B, N, D)
        v = self.project_v(inputs)   # (B, N, D)

        # Large negative for masked positions
        attn_mask = (~mask).float() * -1e9  # (B, N)

        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)  # (B, K, D)

            # Attention: slots query positions
            scale = D ** -0.5
            attn_logits = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, K, N)
            attn_logits = attn_logits + attn_mask.unsqueeze(1)

            # Softmax over slots (competition) then normalize over positions
            attn = F.softmax(attn_logits, dim=1)  # (B, K, N) - slot competition
            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=-1, keepdim=True)  # normalize per slot

            # Weighted mean of values
            updates = torch.bmm(attn, v)  # (B, K, D)

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.num_slots, D),
                slots_prev.reshape(B * self.num_slots, self.d_slot),
            ).reshape(B, self.num_slots, self.d_slot)

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Final attention for output
        q = self.project_q(self.norm_slots(slots))
        attn_logits = torch.bmm(q, k.transpose(1, 2)) * (D ** -0.5)
        attn_logits = attn_logits + attn_mask.unsqueeze(1)
        attn = F.softmax(attn_logits, dim=1)

        return slots, attn


class SlotAttention(nn.Module):
    """Full Slot Attention pipeline for ARC grids.

    Takes one-hot grids, returns object slots.
    """

    def __init__(
        self,
        num_slots: int = 16,
        d_slot: int = 64,
        d_model: int = 64,
        num_iters: int = 3,
    ):
        super().__init__()
        self.encoder = GridEncoder(d_model=d_model)
        self.slot_attn = SlotAttentionModule(
            num_slots=num_slots,
            d_slot=d_slot,
            d_model=d_model,
            num_iters=num_iters,
        )

    def forward(
        self, grid_onehot: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            grid_onehot: (B, NUM_COLORS, H, W)
            mask: (B, H, W) bool
        Returns:
            slots: (B, K, D_slot) object slot representations
            attn: (B, K, N) slot-to-position attention
        """
        features, flat_mask = self.encoder(grid_onehot, mask)
        slots, attn = self.slot_attn(features, flat_mask)
        return slots, attn
