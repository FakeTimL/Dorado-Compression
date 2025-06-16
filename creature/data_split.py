# sample_train_val.py
import os
import random
from pathlib import Path

chunk_dir = Path("training_chunks")
all_chunks = sorted(f.stem for f in chunk_dir.glob("*.npy"))
random.shuffle(all_chunks)

# 25% training, 1000 validation
train_split = int(0.25 * len(all_chunks))
val_size = 1000

train_ids = all_chunks[:train_split]
val_ids = all_chunks[train_split:train_split + val_size]

with open("train.txt", "w") as f:
  for cid in train_ids:
    f.write(cid + "\n")

with open("val.txt", "w") as f:
  for cid in val_ids:
    f.write(cid + "\n")

print(f"Created train.txt ({len(train_ids)} entries) and val.txt ({len(val_ids)} entries)")
