from IPython.display import Markdown, display

model = 'dna_r10.4.1_e8.2_400bps'
version = 'v5.0.0'
bin_path = '/vol/bitbucket/bl1821/dorado-1.0.0-linux-x64/bin'
dorado = f'{bin_path}/dorado'

speeds = ['fast', 'hac', 'sup']
models = {speed: f'{model}_{speed}@{version}' for speed in speeds}
model_paths = {speed: f'{bin_path}/{model}' for speed, model in models.items()}

def tsv_dict(data_dir, speed):
  with open(f'{data_dir}/{speed}.tsv', 'r') as f:
    a = f.readlines()
  return dict(zip(a[0].split(),a[1].split()))

def make_table(data, markdown=True):
  # row_headers = sorted({key for inner in data.values() for key in inner})
  row_headers = [
    "alignment_accuracy", "alignment_identity", 
    "alignment_genome_start", "alignment_genome_end", "alignment_strand_start", "alignment_strand_end", "alignment_strand_coverage",
    "alignment_length", "alignment_num_aligned", "alignment_num_correct", "alignment_num_deletions", "alignment_num_insertions", "alignment_num_substitutions",
    ]

  # Build markdown table
  header_row = f"| Metric | {' | '.join(data.keys())} |"
  separator_row = f"|--------|{'|'.join(['--------'] * len(data))}|"
  data_rows = [
    f"| {key} | {' | '.join(str(data[name].get(key, '')) for name in data)} |"
    for key in row_headers
  ]

  table_md = "\n".join([header_row, separator_row] + data_rows)

  if markdown:
    display(Markdown(table_md))
  else:
    print(table_md)
