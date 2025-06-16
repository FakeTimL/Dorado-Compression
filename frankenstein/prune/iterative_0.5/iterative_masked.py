import torch
import torch.nn.utils.prune as prune
from pathlib import Path
from bonito.util import load_model
import shutil
import sys
import pickle

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 iterative_masked.py <model_dir> <sparsity> <save_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    target_sparsity = float(sys.argv[2])
    save_dir = Path(sys.argv[3])

    if not (0.0 <= target_sparsity <= 1.0):
        print("Error: Sparsity must be between 0.0 and 1.0")
        sys.exit(1)

    config_path = model_dir / "config.toml"
    if not config_path.exists():
        print(f"Error: config.toml not found in {model_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Iterative pruning to target sparsity {target_sparsity:.2f}")

    # Load model
    model = load_model(str(model_dir), device=device)

    # Load previous masks if they exist
    prev_masks = {}
    masks_path = model_dir / "masks.pkl"
    if masks_path.exists():
        with open(masks_path, "rb") as f:
            prev_masks = pickle.load(f)
        print(f"Loaded previous masks from {masks_path}")
    else:
        print("No previous masks found. Starting from dense model.")

    # Apply previous masks to model
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in prev_masks:
                mask = prev_masks[name].to(device)
                if mask.shape == param.shape:
                    param.data.mul_(mask)
                else:
                    print(f"Warning: Mask shape {mask.shape} doesn't match parameter shape {param.shape} for {name}")

    # Compute current sparsity
    total_weights = 0
    total_zero = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            total_weights += weight.numel()
            total_zero += weight.numel() - weight.count_nonzero().item()
    current_sparsity = total_zero / total_weights
    print(f"Current sparsity: {current_sparsity:.4f}")

    if target_sparsity < current_sparsity:
        print(f"Error: Target sparsity {target_sparsity} is less than current sparsity {current_sparsity:.4f}")
        sys.exit(1)

    effective_amount = (target_sparsity - current_sparsity) / (1 - current_sparsity)
    print(f"Applying effective pruning amount: {effective_amount:.4f}")

    # Collect parameters to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))

    # Apply global pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=effective_amount,
    )

    # Save new masks
    new_masks = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight_mask"):
            new_masks[name + ".weight"] = module.weight_mask.clone().detach().cpu()

    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "weights_1.tar")
    with open(save_dir / "masks.pkl", "wb") as f:
        pickle.dump(new_masks, f)
    shutil.copy(config_path, save_dir / "config.toml")

    print(f"Saved pruned model and updated masks to {save_dir}")

if __name__ == "__main__":
    main()
