import os
import pod5 as p5
import pysam
import numpy as np
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
pod5_file = "/vol/bitbucket/bl1821/training_data/PAW79146_c6420e1f_4af91fd6_0.pod5"
bam_file = "/vol/bitbucket/bl1821/training_data/calls.sorted.bam"
output_dir = "training_chunks"

chunksize = 8192         # reduced from 12288
overlap = 512            # reduced from 600
stride = chunksize - overlap

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# STEP 1: Build pod5 read ID map (prefix → full ID)
# -----------------------------
print("Indexing pod5 read IDs...")
pod5_reader = p5.Reader(pod5_file)
pod5_read_ids = {}
for read in tqdm(pod5_reader.reads(), total=pod5_reader.num_reads):
  short_id = str(read.read_id)[:20]
  pod5_read_ids[short_id] = str(read.read_id)

# -----------------------------
# STEP 2: Map matching BAM read IDs to sequences
# -----------------------------
print("Scanning BAM alignments...")
id_to_label = {}

bam = pysam.AlignmentFile(bam_file, "rb")
for aln in tqdm(bam.fetch(until_eof=True)):
  bam_id = aln.query_name[:20]
  if bam_id in pod5_read_ids and bam_id not in id_to_label:
    seq = aln.query_sequence
    if seq:
      full_id = pod5_read_ids[bam_id]
      id_to_label[full_id] = seq

print(f"Mapped {len(id_to_label)} pod5 reads to basecall sequences")

# -----------------------------
# STEP 3: Chunk signals and save labels
# -----------------------------
print("Extracting and saving chunks...")
pod5_reader = p5.Reader(pod5_file)
num_chunks = 0
short_reads = 0

for read in tqdm(pod5_reader.reads(), total=pod5_reader.num_reads):
  read_id = str(read.read_id)
  if read_id not in id_to_label:
    continue

  signal = read.signal
  label = id_to_label[read_id]

  if len(signal) < chunksize:
    short_reads += 1
    continue

  for i in range(0, len(signal) - chunksize + 1, stride):
    chunk = signal[i:i+chunksize]
    chunk_file = os.path.join(output_dir, f"{read_id}_{i}.npy")
    label_file = chunk_file.replace(".npy", ".label.txt")

    np.save(chunk_file, chunk)
    with open(label_file, "w") as f:
      f.write(label)

    num_chunks += 1

print(f"\nDone. Extracted {num_chunks} chunks to: {output_dir}")
print(f"Skipped {short_reads} reads that were shorter than {chunksize} samples")
