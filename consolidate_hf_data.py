"""
Consolidate scattered HF dataset repos into a single clean repo.

  formatted/ ← YUGOROU/amd-voice-sft-data  (train + val splits)
  rewritten/ ← YUGOROU/amd-voice-sft-data/rewritten/

Run:
  python3 consolidate_hf_data.py
  (uses cached huggingface-cli login)
"""
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi

NEW_REPO = "YUGOROU/lumi-data"
SRC_REPO = "YUGOROU/amd-voice-sft-data"

api = HfApi()

# ── Create new repo ───────────────────────────────────────────────────────────
api.create_repo(NEW_REPO, repo_type="dataset", private=True, exist_ok=True)
print(f"Repo ready: {NEW_REPO}")

# ── Copy formatted/ (train + val) ────────────────────────────────────────────
print(f"\nLoading formatted data from {SRC_REPO} ...")
train_ds = load_dataset(SRC_REPO, data_files="train.jsonl", split="train")
val_ds   = load_dataset(SRC_REPO, data_files="val.jsonl",   split="train")
all_splits = DatasetDict({"train": train_ds, "val": val_ds})
print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

all_splits.push_to_hub(
    NEW_REPO,
    data_dir="formatted",
    private=True,
    commit_message="Add formatted/ (ChatML preprocessed, 3 public datasets)",
)
for split, ds in all_splits.items():
    print(f"  ✓ formatted/{split}: {len(ds):,} samples")

# ── Copy rewritten/ (raw parquet upload to avoid schema mismatch) ─────────────
import tempfile, os
from huggingface_hub import snapshot_download

print(f"\nDownloading rewritten/ from {SRC_REPO} ...")
tmpdir = tempfile.mkdtemp()
snapshot_download(
    repo_id=SRC_REPO,
    repo_type="dataset",
    allow_patterns=["rewritten/**"],
    local_dir=tmpdir,
)
rw_dir = os.path.join(tmpdir, "rewritten")
assert os.path.isdir(rw_dir), "rewritten/ not found"

api.upload_folder(
    folder_path=rw_dir,
    path_in_repo="rewritten",
    repo_id=NEW_REPO,
    repo_type="dataset",
    commit_message="Add rewritten/ (Layer 1 EQ-Matrix rewrite, 12,375 samples)",
)
print(f"  ✓ rewritten/ uploaded")

print(f"""
Done.
  {NEW_REPO}/formatted/  ← uploaded
  {NEW_REPO}/rewritten/  ← uploaded
  {NEW_REPO}/filtered/   ← upload from Colab (see snippet below)

Colab snippet:
  Dataset.from_list(valid_samples).push_to_hub(
      "YUGOROU/lumi-data",
      data_dir="filtered",
      private=True,
  )
""")
