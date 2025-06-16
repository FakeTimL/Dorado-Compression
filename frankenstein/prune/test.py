#!/usr/bin/env python3
import torch
from pathlib import Path
from bonito.util import load_model
import sys
import torch.nn.utils.prune as prune

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 inspect_sparse_model.py <sparse_model_dir>"); sys.exit(1)

    sparse_dir = Path(sys.argv[1])
    device = "cpu"
    # load model structure & state dict
    model = load_model(str(sparse_dir), device=device)
    sd    = torch.load(sparse_dir / "sparse_weights.tar", map_location=device)
    model.load_state_dict(sd, strict=False)

    print(f"Inspecting sparse model in {sparse_dir}\n")
    for name, module in model.named_modules():
        mask_key = f"{name}.weight_mask"
        if mask_key in sd:
            mask = sd[mask_key]
            total = mask.numel()
            nz    = int(mask.sum().item())
            print(f"Layer: {name:30s} | Mask nonzeros: {nz:6d}/{total:6d}  ({nz/total:.2%})")

if __name__ == "__main__":
    main()
