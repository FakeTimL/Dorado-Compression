import torch
import torch.nn.utils.prune as prune
from pathlib import Path
from bonito.util import load_model
import shutil
import sys
import pickle

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 one_shot.py <model_dir> <sparsity> <save_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    sparsity = float(sys.argv[2])
    save_dir = Path(sys.argv[3])

    if not (0.0 <= sparsity <= 1.0):
        print("Error: Sparsity must be between 0.0 and 1.0")
        sys.exit(1)

    config_path = model_dir / "config.toml"
    if not config_path.exists():
        print(f"Error: config.toml not found in {model_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pruning model at sparsity {sparsity:.2f}")

    # Load model
    model = load_model(str(model_dir), device=device)

    # Collect parameters to prune (only weights, not biases)
    masks = {}
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Only prune weight parameters
            parameters_to_prune.append((module, "weight"))
            masks[name + ".weight"] = None  # Initialize mask storage

    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    # Save masks for weights only
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight_mask"):
            # Save mask with full parameter shape
            masks[name + ".weight"] = module.weight_mask.clone().detach().cpu()

    # Remove pruning reparameterization and save clean state dict
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")

    # Save pruned model and masks
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "weights_1.tar")
    with open(save_dir / "masks.pkl", "wb") as f:
        pickle.dump(masks, f)
    shutil.copy(config_path, save_dir / "config.toml")

    print(f"Saved pruned model and masks to {save_dir}")

if __name__ == "__main__":
    main()