"""Type Classifier: classify observed transformations into MicroOpTypes.

Given input/output slot pairs, classify the transformation along
the 4 lattice dimensions: topology, color, geometry, cardinality.

This bridges the neural (Slot Attention) and symbolic (Type Lattice) worlds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .type_lattice import (
    MicroOpType,
    TopologyChange,
    ColorMap,
    GeometryOp,
    CardinalityDelta,
)


NUM_TOPOLOGY = len(TopologyChange) - 1   # exclude ANY
NUM_COLOR = len(ColorMap) - 1
NUM_GEOMETRY = len(GeometryOp) - 1
NUM_CARDINALITY = len(CardinalityDelta) - 1


class TransformationClassifier(nn.Module):
    """Classify the transformation between input and output slot pairs.

    Takes concatenated [input_slot, output_slot] and predicts
    the 4-dimensional type.
    """

    def __init__(self, d_slot: int = 64, d_hidden: int = 128):
        super().__init__()

        d_in = d_slot * 2  # concatenated input + output

        self.shared = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )

        # Per-dimension classification heads
        self.topology_head = nn.Linear(d_hidden, NUM_TOPOLOGY)
        self.color_head = nn.Linear(d_hidden, NUM_COLOR)
        self.geometry_head = nn.Linear(d_hidden, NUM_GEOMETRY)
        self.cardinality_head = nn.Linear(d_hidden, NUM_CARDINALITY)

    def forward(
        self, input_slots: torch.Tensor, output_slots: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_slots: (B, K, D_slot)
            output_slots: (B, K, D_slot)
        Returns:
            dict with logits per dimension, each (B, K, num_classes)
        """
        B, K, D = input_slots.shape
        x = torch.cat([input_slots, output_slots], dim=-1)  # (B, K, 2D)
        x = x.reshape(B * K, -1)
        h = self.shared(x)  # (B*K, d_hidden)

        return {
            "topology": self.topology_head(h).reshape(B, K, -1),
            "color": self.color_head(h).reshape(B, K, -1),
            "geometry": self.geometry_head(h).reshape(B, K, -1),
            "cardinality": self.cardinality_head(h).reshape(B, K, -1),
        }

    def predict_types(
        self, input_slots: torch.Tensor, output_slots: torch.Tensor
    ) -> list[list[MicroOpType]]:
        """Predict MicroOpType for each slot in each batch element.

        Returns list[list[MicroOpType]]: outer = batch, inner = slots.
        """
        logits = self.forward(input_slots, output_slots)

        B, K, _ = logits["topology"].shape
        results = []

        for b in range(B):
            slot_types = []
            for k in range(K):
                t = logits["topology"][b, k].argmax().item()
                c = logits["color"][b, k].argmax().item()
                g = logits["geometry"][b, k].argmax().item()
                d = logits["cardinality"][b, k].argmax().item()
                slot_types.append(MicroOpType(
                    topology=TopologyChange(t),
                    color=ColorMap(c),
                    geometry=GeometryOp(g),
                    cardinality=CardinalityDelta(d),
                ))
            results.append(slot_types)

        return results
