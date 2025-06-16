import os
import shutil
import torch

def one_shot_prune_tensor(tensor, sparsity):
  """Returns a pruned version of the tensor using global L1 pruning."""
  flat = tensor.view(-1)
  threshold = torch.quantile(flat.abs(), sparsity)
  mask = (tensor.abs() >= threshold).float()
  return tensor * mask

def should_prune_tensor(fname):
  """Returns True if this .tensor file should be pruned."""
  return (
    fname.endswith(".tensor") and
    "weight" in fname and  # skip bias/norm/deepnorm
    not any(x in fname for x in ["norm", "bias", "deepnorm", "alpha", "beta"])
  )
  
def one_shot_prune_model(src_dir, dst_dir, sparsity=0.9):
  os.makedirs(dst_dir, exist_ok=True)

  for fname in os.listdir(src_dir):
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    if fname.endswith(".tensor"):
      try:
        mod = torch.jit.load(src_path)
        state = mod.state_dict()

        if len(state) != 1:
          print(f"Skipping {fname}: expected exactly 1 parameter, got {len(state)}")
          continue

        param_name, tensor = next(iter(state.items()))
        param_count = tensor.numel()

        if should_prune_tensor(fname) and tensor.dim() > 1:
          pruned = one_shot_prune_tensor(tensor, sparsity)
          mod.load_state_dict({param_name: pruned})
          torch.jit.save(mod, dst_path)
          # print(f"Pruned {fname} ({param_count} params, {sparsity*100:.1f}% sparsity)")
        else:
          shutil.copy2(src_path, dst_path)
          # print(f"Skipped {fname} ({param_count} params)")

      except Exception as e:
        print(f"Failed to process {fname}: {e}")

    elif fname == "config.toml":
      shutil.copy2(src_path, dst_path)
      
def iterative_prune_model(src_dir, dest_dir, sparsity_per_step=0.1, steps=5):
  """
  Iteratively prunes the model using one-shot unstructured pruning for each step.
  Saves a full model copy at each step to dest_dir/step_0, step_1, ..., step_N.
  """
  assert 0 < sparsity_per_step < 1
  os.makedirs(dest_dir, exist_ok=True)

  # Load all tensors once into memory
  model_tensors = {}
  for fname in os.listdir(src_dir):
    if fname.endswith(".tensor"):
      mod = torch.jit.load(os.path.join(src_dir, fname))
      state = mod.state_dict()
      if len(state) == 1:
        model_tensors[fname] = (mod, next(iter(state.items())))

  # Save initial unpruned model
  step_dir = os.path.join(dest_dir, "step_0")
  os.makedirs(step_dir, exist_ok=True)
  for fname, (mod, _) in model_tensors.items():
    torch.jit.save(mod, os.path.join(step_dir, fname))
  shutil.copy2(os.path.join(src_dir, "config.toml"), os.path.join(step_dir, "config.toml"))
  print(f"Saved initial model to {step_dir}")

  # Begin iterative pruning
  for step in range(1, steps + 1):
    print(f"\n🔁 Pruning step {step}/{steps} (pruning {sparsity_per_step:.0%} of remaining weights)")
    for fname, (mod, (param_name, tensor)) in model_tensors.items():
      if should_prune_tensor(fname) and tensor.dim() > 1:
        pruned_tensor = one_shot_prune_tensor(tensor, sparsity_per_step)
        mod.load_state_dict({param_name: pruned_tensor})
        model_tensors[fname] = (mod, (param_name, pruned_tensor))

    # Save current step
    step_dir = os.path.join(dest_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    for fname, (mod, _) in model_tensors.items():
      torch.jit.save(mod, os.path.join(step_dir, fname))
    shutil.copy2(os.path.join(src_dir, "config.toml"), os.path.join(step_dir, "config.toml"))
    print(f"Saved pruned model to {step_dir}")