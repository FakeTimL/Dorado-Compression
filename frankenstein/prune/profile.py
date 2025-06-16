import torch
from pathlib import Path
from bonito.util import load_model
import sys

def count_parameters(model):
    total_params = 0
    nonzero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += param.nonzero().size(0)

    return total_params, nonzero_params

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 count_params.py <model_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    config_path = model_dir / "config.toml"
    weights_path = model_dir / "weights_1.tar"

    if not config_path.exists() or not weights_path.exists():
        print(f"Error: Model files not found in {model_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(model_dir), device=device)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    total, nonzero = count_parameters(model)
    sparsity = 1.0 - (nonzero / total)

    print(f"Model: {model_dir}")
    print(f"Total parameters:    {total:,}")
    print(f"Nonzero parameters:  {nonzero:,}")
    print(f"Sparsity:            {sparsity:.2%}")

if __name__ == "__main__":
    main()
