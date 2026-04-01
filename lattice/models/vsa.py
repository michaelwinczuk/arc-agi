"""Vector Symbolic Architecture (VSA) for ARC transformation encoding.

Encodes object slots as high-dimensional binary hypervectors.
Transformation = XOR of input and output slot vectors.
Consensus across demo pairs via majority vote.

Based on Kanerva's Hyperdimensional Computing.
At d=10,000: false positive rate < 10^-6.

Target: ~1ms per projection, ~0.1ms per XOR delta.
"""

import torch
import torch.nn as nn


class VSAEncoder(nn.Module):
    """Project slot representations into binary hypervector space.

    Learned projection from continuous slot space to binary VSA space.
    Uses straight-through estimator for binarization during training.
    """

    def __init__(self, d_slot: int = 64, d_vsa: int = 10000):
        super().__init__()
        self.d_vsa = d_vsa

        # Learned projection to VSA space
        self.project = nn.Sequential(
            nn.Linear(d_slot, 512),
            nn.ReLU(),
            nn.Linear(512, d_vsa),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: (B, K, D_slot) continuous slot representations
        Returns:
            (B, K, D_vsa) binary hypervectors {0, 1}
        """
        logits = self.project(slots)  # (B, K, D_vsa)

        if self.training:
            # Straight-through estimator: binary forward, continuous backward
            binary = (logits > 0).float()
            return binary + logits - logits.detach()  # STE
        else:
            return (logits > 0).float()


class VSAOperations:
    """Core VSA operations on binary hypervectors.

    All operations are O(d) where d = dimensionality.
    No learned parameters - pure algebraic operations.
    """

    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """XOR binding - creates association between two vectors.
        bind(a, b) is dissimilar to both a and b.
        bind(bind(a, b), b) ≈ a (self-inverse).

        For float {0,1} tensors, XOR = (a + b) mod 2 = a + b - 2*a*b
        """
        return a + b - 2 * a * b

    @staticmethod
    def bundle(vectors: torch.Tensor) -> torch.Tensor:
        """Majority vote bundling - creates superposition.

        Args:
            vectors: (N, ..., D) multiple vectors to bundle
        Returns:
            (..., D) bundled vector (majority vote per dimension)
        """
        return (vectors.mean(dim=0) > 0.5).float()

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity in VSA space.
        For binary vectors, this correlates with Hamming distance.

        Args:
            a: (..., D) query vector
            b: (..., D) candidate vector(s)
        Returns:
            (...) similarity scores in [-1, 1]
        """
        # Normalize to {-1, +1} for cosine
        a_norm = 2 * a - 1
        b_norm = 2 * b - 1
        return (a_norm * b_norm).mean(dim=-1)

    @staticmethod
    def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Hamming distance (number of differing bits).

        Args:
            a, b: (..., D) binary vectors
        Returns:
            (...) integer distances
        """
        return (a != b).float().sum(dim=-1)


class DeltaExtractor(nn.Module):
    """Extract transformation deltas between input and output slot sets.

    Given slots from input grid and output grid, compute the
    transformation as a VSA delta vector.

    The delta captures what changed: XOR(input_slots, output_slots).
    """

    def __init__(self, d_slot: int = 64, d_vsa: int = 10000, num_slots: int = 16):
        super().__init__()
        self.vsa_encoder = VSAEncoder(d_slot=d_slot, d_vsa=d_vsa)
        self.ops = VSAOperations()
        self.num_slots = num_slots

        # Slot-level matching: learn to align input slots to output slots
        self.slot_matcher = nn.Sequential(
            nn.Linear(d_slot * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def match_slots(
        self, input_slots: torch.Tensor, output_slots: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft assignment matrix between input and output slots.

        Args:
            input_slots: (B, K, D_slot)
            output_slots: (B, K, D_slot)
        Returns:
            (B, K_in, K_out) assignment probabilities
        """
        B, K, D = input_slots.shape

        # All pairs: (B, K, K, 2D)
        inp_exp = input_slots.unsqueeze(2).expand(B, K, K, D)
        out_exp = output_slots.unsqueeze(1).expand(B, K, K, D)
        pairs = torch.cat([inp_exp, out_exp], dim=-1)

        scores = self.slot_matcher(pairs).squeeze(-1)  # (B, K, K)
        return torch.softmax(scores, dim=-1)  # normalize over output slots

    def forward(
        self,
        input_slots: torch.Tensor,
        output_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute transformation delta between input and output.

        Args:
            input_slots: (B, K, D_slot)
            output_slots: (B, K, D_slot)
        Returns:
            delta: (B, D_vsa) transformation vector
            input_vsa: (B, K, D_vsa) encoded input slots
        """
        # Project to VSA space
        input_vsa = self.vsa_encoder(input_slots)    # (B, K, D_vsa)
        output_vsa = self.vsa_encoder(output_slots)  # (B, K, D_vsa)

        # Bundle all slots per grid into single holographic vector
        input_bundled = self.ops.bundle(input_vsa.transpose(0, 1))   # (B, D_vsa)
        output_bundled = self.ops.bundle(output_vsa.transpose(0, 1)) # (B, D_vsa)

        # Delta = XOR(input, output) - captures the transformation
        delta = self.ops.bind(input_bundled, output_bundled)  # (B, D_vsa)

        return delta, input_vsa


class ConsensusBuilder:
    """Build consensus transformation from multiple demo pairs.

    Uses majority vote across demo pair deltas.
    If Hamming distance between any delta and consensus > threshold,
    the task may require the Type Lattice fallback.
    """

    def __init__(self, d_vsa: int = 10000, consistency_threshold: int = 800):
        self.ops = VSAOperations()
        self.consistency_threshold = consistency_threshold

    def build_consensus(
        self, deltas: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """
        Args:
            deltas: (N, D_vsa) transformation deltas from N demo pairs
        Returns:
            consensus: (D_vsa,) majority-vote consensus delta
            is_consistent: True if all deltas are within threshold of consensus
        """
        consensus = self.ops.bundle(deltas)  # (D_vsa,)

        # Check consistency
        distances = self.ops.hamming_distance(
            deltas, consensus.unsqueeze(0)
        )  # (N,)

        is_consistent = bool((distances <= self.consistency_threshold).all())

        return consensus, is_consistent

    def apply_delta(
        self, test_input_vsa: torch.Tensor, consensus: torch.Tensor
    ) -> torch.Tensor:
        """Apply consensus transformation to test input.

        Args:
            test_input_vsa: (K, D_vsa) or (B, K, D_vsa) test input slot vectors
            consensus: (D_vsa,) consensus transformation
        Returns:
            predicted_output_vsa: same shape as input
        """
        return self.ops.bind(test_input_vsa, consensus)
