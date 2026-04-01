"""Cross-Attention Decoder: learn input→output transformation from demos.

The original SlotDecoder just decodes slots to a grid independently.
This decoder adds a cross-attention mechanism that conditions on:
1. Demo input/output slot pairs (what transformation looks like)
2. Test input slots (what to transform)

Architecture:
  demo_pairs → TransformationEncoder → transformation embedding
  test_input_slots + transformation_embedding → CrossAttentionDecoder → output grid

This is the key module that makes the neural pipeline actually work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.arc_dataset import NUM_COLORS, MAX_GRID_SIZE


class TransformationEncoder(nn.Module):
    """Encode the transformation pattern from demo input/output pairs.

    Processes each demo pair independently, then aggregates into a single
    transformation embedding that captures *what changes* between input and output.
    """

    def __init__(self, d_slot: int = 64, d_transform: int = 128):
        super().__init__()
        self.d_transform = d_transform

        # Encode a single input-output slot pair
        self.pair_encoder = nn.Sequential(
            nn.Linear(d_slot * 2, d_transform),
            nn.ReLU(),
            nn.Linear(d_transform, d_transform),
            nn.ReLU(),
        )

        # Self-attention to find consistent transformation across slot pairs
        self.self_attn = nn.MultiheadAttention(
            d_transform, num_heads=4, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_transform)

        # Aggregate across demo pairs
        self.demo_attn = nn.MultiheadAttention(
            d_transform, num_heads=4, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_transform)

        # Final projection
        self.project = nn.Sequential(
            nn.Linear(d_transform, d_transform),
            nn.ReLU(),
        )

    def encode_pair(
        self, input_slots: torch.Tensor, output_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Encode transformation from a single demo pair.

        Args:
            input_slots: (1, K, D_slot) input object slots
            output_slots: (1, K, D_slot) output object slots
        Returns:
            (1, K, D_transform) per-slot transformation features
        """
        # Concatenate input and output per slot
        paired = torch.cat([input_slots, output_slots], dim=-1)  # (1, K, 2*D_slot)
        encoded = self.pair_encoder(paired)  # (1, K, D_transform)

        # Self-attention across slots to capture inter-slot relationships
        attended, _ = self.self_attn(encoded, encoded, encoded)
        encoded = self.norm1(encoded + attended)

        return encoded

    def forward(
        self, pair_encodings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate transformation encodings across demo pairs.

        Args:
            pair_encodings: list of (1, K, D_transform) from each demo pair
        Returns:
            (1, K, D_transform) aggregated transformation embedding
        """
        if len(pair_encodings) == 1:
            return self.project(pair_encodings[0])

        # Stack all demo pair encodings: (N_demos, K, D_transform)
        stacked = torch.cat(pair_encodings, dim=0)  # (N, K, D)

        # Use first pair as query, attend to all pairs
        query = pair_encodings[0]  # (1, K, D)

        # Reshape for cross-attention: flatten slots across demos
        N, K, D = stacked.shape
        kv = stacked.reshape(1, N * K, D)  # (1, N*K, D)

        attended, _ = self.demo_attn(query, kv, kv)
        result = self.norm2(query + attended)

        return self.project(result)


class CrossAttentionDecoder(nn.Module):
    """Decode output grid conditioned on transformation + input slots.

    Takes:
    - test input slots (what to transform)
    - transformation embedding (how to transform)
    - target output size (H, W)

    Produces:
    - (B, NUM_COLORS, H, W) color logits
    """

    def __init__(
        self,
        d_slot: int = 64,
        d_transform: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input slots to decoder space
        self.input_project = nn.Linear(d_slot, d_model)

        # Project transformation embedding to decoder space
        self.transform_project = nn.Linear(d_transform, d_model)

        # Cross-attention: output positions attend to transformed input slots
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(d_model)

        # Self-attention over output positions
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True,
        )
        self.norm_self = nn.LayerNorm(d_model)

        # Positional encoding for output grid
        self.pos_embed = nn.Parameter(
            torch.randn(1, MAX_GRID_SIZE * MAX_GRID_SIZE, d_model) * 0.02
        )

        # MLP + output head
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_mlp = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, NUM_COLORS)

    def forward(
        self,
        input_slots: torch.Tensor,
        transform_embed: torch.Tensor,
        input_attn: torch.Tensor,
        out_h: int,
        out_w: int,
    ) -> torch.Tensor:
        """
        Args:
            input_slots: (B, K, D_slot) test input object slots
            transform_embed: (B, K, D_transform) transformation embedding
            input_attn: (B, K, N_in) slot attention map from input
            out_h, out_w: output grid dimensions
        Returns:
            (B, NUM_COLORS, out_h, out_w) color logits
        """
        B = input_slots.shape[0]
        N_out = out_h * out_w

        # Build transformed key-value from input slots + transformation
        inp_proj = self.input_project(input_slots)       # (B, K, D_model)
        trans_proj = self.transform_project(transform_embed)  # (B, K, D_model)

        # Combine: element-wise add (transformation modifies input representation)
        kv = inp_proj + trans_proj  # (B, K, D_model)

        # Output position queries with positional encoding
        queries = self.pos_embed[:, :N_out, :].expand(B, -1, -1)  # (B, N_out, D_model)

        # Cross-attention: output positions attend to transformed input slots
        attended, _ = self.cross_attn(queries, kv, kv)
        queries = self.norm_cross(queries + attended)

        # Self-attention over output positions (spatial coherence)
        self_attended, _ = self.self_attn(queries, queries, queries)
        queries = self.norm_self(queries + self_attended)

        # MLP
        queries = queries + self.mlp(self.norm_mlp(queries))

        # Project to color logits
        logits = self.output_head(queries)  # (B, N_out, NUM_COLORS)
        logits = logits.transpose(1, 2)     # (B, NUM_COLORS, N_out)
        logits = logits.view(B, NUM_COLORS, out_h, out_w)

        return logits

    def decode_grid(
        self,
        input_slots: torch.Tensor,
        transform_embed: torch.Tensor,
        input_attn: torch.Tensor,
        out_h: int,
        out_w: int,
    ) -> torch.Tensor:
        """Decode to discrete grid."""
        logits = self.forward(input_slots, transform_embed, input_attn, out_h, out_w)
        return logits.argmax(dim=1)  # (B, H, W)
