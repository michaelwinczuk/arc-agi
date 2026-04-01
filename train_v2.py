"""V2 Training: includes cross-attention decoder.

Run after v1 finishes to fine-tune with the full architecture.
Can optionally warm-start from v1 checkpoint.

Usage:
    python -u train_v2.py                    # fresh start
    python -u train_v2.py --warmstart        # load v1 weights for shared modules
"""

import sys
import os
import argparse
sys.path.insert(0, ".")
os.environ["PYTHONUNBUFFERED"] = "1"

from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmstart", action="store_true",
                        help="Load shared weights from v1 checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    print("Starting V2 training (with cross-attention decoder)...", flush=True)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    from lattice.train import LatticeTrainer, train

    if args.warmstart:
        v1_path = Path("checkpoints/best.pt")
        if v1_path.exists():
            print(f"Warm-starting from {v1_path}", flush=True)
            # Load v1 checkpoint to get shared weights
            ckpt = torch.load(v1_path, map_location=device, weights_only=False)
            v1_state = ckpt.get("model_state_dict", ckpt)

            # Create v2 model
            model = LatticeTrainer(
                num_slots=16, d_slot=64, d_model=64, d_vsa=10000,
            )

            # Load matching weights (v1 has slot_attention, decoder, delta_extractor)
            own_state = model.state_dict()
            loaded = 0
            for name, param in v1_state.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded += 1
            print(f"Loaded {loaded}/{len(own_state)} params from v1", flush=True)
        else:
            print(f"No v1 checkpoint at {v1_path}, training from scratch", flush=True)

    train(
        data_dir=Path("data/ARC-AGI-2/data/training"),
        epochs=args.epochs,
        lr=2e-4,  # slightly lower LR for fine-tuning
        device_str="cuda" if torch.cuda.is_available() else "cpu",
        num_slots=16,
        d_slot=64,
        d_model=64,
        d_vsa=10000,
        save_every=10,
        save_dir="checkpoints_v2",
        augment=True,
    )


if __name__ == "__main__":
    main()
