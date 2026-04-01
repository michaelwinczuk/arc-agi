"""Training launch script for Alienware RTX 5070 (8GB VRAM)."""
import sys
import os
sys.path.insert(0, ".")

# Force flush output
os.environ["PYTHONUNBUFFERED"] = "1"

from pathlib import Path

def main():
    print("Starting training...", flush=True)

    import torch
    print(f"CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM: {mem / 1e9:.1f}GB", flush=True)

    from lattice.train import train

    # Tuned for 8GB VRAM RTX 5070
    train(
        data_dir=Path("data/ARC-AGI-2/data/training"),
        epochs=100,
        lr=3e-4,
        device_str="cuda",
        num_slots=16,
        d_slot=64,
        d_model=64,
        d_vsa=10000,     # full size, only 0.1GB VRAM
        save_every=10,
        save_dir="checkpoints",
        augment=True,
    )

if __name__ == "__main__":
    main()
