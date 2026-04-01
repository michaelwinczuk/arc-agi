"""Test-Time Library: DreamCoder-style micro-op caching.

As puzzles are solved during test time, their micro-ops are cached.
Puzzle N benefits from solutions to puzzles 1..N-1.

Key insight: discrete fragments (microseconds) not gradient updates (minutes).
This is our speed advantage over NVARC's LoRA fine-tuning.
"""

import torch
from dataclasses import dataclass, field

from .type_lattice import MicroOpType, TypeLattice, types_composable
from .vsa import VSAOperations


@dataclass
class LibraryEntry:
    """A cached micro-op from a solved puzzle."""
    op_type: MicroOpType
    vsa_delta: torch.Tensor      # (D_vsa,) the transformation vector
    source_task_id: str           # which puzzle it came from
    confidence: float             # how well it worked (0-1)
    composition_depth: int = 1    # 1 = primitive, >1 = composed


class TestTimeLibrary:
    """Accumulates micro-ops as puzzles are solved.

    Provides fast lookup by type or VSA similarity.
    """

    def __init__(self, d_vsa: int = 10000):
        self.entries: list[LibraryEntry] = []
        self.lattice = TypeLattice()
        self.ops = VSAOperations()
        self.d_vsa = d_vsa

        # VSA index: stack of all deltas for fast batch similarity
        self._vsa_index: list[torch.Tensor] = []
        self._index_dirty = True

    def add(self, entry: LibraryEntry):
        """Add a solved micro-op to the library."""
        self.entries.append(entry)
        self.lattice.register(entry.op_type, {
            "entry_idx": len(self.entries) - 1,
            "task_id": entry.source_task_id,
        })
        self._vsa_index.append(entry.vsa_delta)
        self._index_dirty = True

    def lookup_by_type(self, op_type: MicroOpType) -> list[LibraryEntry]:
        """Find entries matching or composable with the given type."""
        results = []
        for entry in self.entries:
            if types_composable(op_type, entry.op_type):
                results.append(entry)
        return results

    def lookup_by_similarity(
        self, query_delta: torch.Tensor, top_k: int = 5
    ) -> list[tuple[LibraryEntry, float]]:
        """Find entries most similar to query delta in VSA space.

        Args:
            query_delta: (D_vsa,) query transformation vector
            top_k: number of results
        Returns:
            List of (entry, similarity_score) tuples
        """
        if not self._vsa_index:
            return []

        # Stack all library deltas
        index = torch.stack(self._vsa_index)  # (N, D_vsa)
        sims = self.ops.similarity(query_delta.unsqueeze(0), index)  # (N,)

        k = min(top_k, len(self.entries))
        top_vals, top_idxs = sims.topk(k)

        return [
            (self.entries[idx.item()], val.item())
            for val, idx in zip(top_vals, top_idxs)
        ]

    def try_compose(
        self, needed_type: MicroOpType, max_depth: int = 3
    ) -> list[list[LibraryEntry]]:
        """Try to compose library entries to match a needed type.

        Returns possible composition chains.
        """
        candidates = self.lookup_by_type(needed_type)
        if not candidates:
            return []

        candidate_types = [e.op_type for e in candidates]
        chains = self.lattice.build_composition_chain(
            candidate_types, max_depth=max_depth
        )

        # Map back to entries
        type_to_entries = {}
        for entry in candidates:
            key = entry.op_type.to_key()
            type_to_entries.setdefault(key, []).append(entry)

        result = []
        for chain in chains:
            entry_chain = []
            valid = True
            for op_type in chain:
                entries = type_to_entries.get(op_type.to_key(), [])
                if entries:
                    # Pick highest confidence
                    best = max(entries, key=lambda e: e.confidence)
                    entry_chain.append(best)
                else:
                    valid = False
                    break
            if valid:
                result.append(entry_chain)

        return result

    @property
    def size(self) -> int:
        return len(self.entries)

    def stats(self) -> dict:
        return {
            "num_entries": self.size,
            "num_types": self.lattice.size,
            "avg_confidence": (
                sum(e.confidence for e in self.entries) / max(self.size, 1)
            ),
        }
