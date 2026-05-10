"""
Migrate HF dataset repos:
  1. Copy rewritten/ from amd-voice-sft-dataset → amd-voice-sft-data
  2. Delete filtered/ from amd-voice-sft-dataset (bad Layer 2 v1 results)

Run:
  HF_TOKEN=hf_xxx python3 migrate_hf_datasets.py
  or in Colab:
  ! HF_TOKEN="hf_xxx" python3 migrate_hf_datasets.py
"""
import os
import tempfile

from huggingface_hub import CommitOperationDelete, HfApi, snapshot_download

TOKEN  = os.environ["HF_TOKEN"]
SRC    = "YUGOROU/amd-voice-sft-dataset"
DST    = "YUGOROU/amd-voice-sft-data"

api = HfApi(token=TOKEN)

# ── Step 1: copy rewritten/ ──────────────────────────────────────────────────
print(f"Downloading rewritten/ from {SRC} ...")
tmpdir = tempfile.mkdtemp()
snapshot_download(
    repo_id=SRC,
    repo_type="dataset",
    allow_patterns=["rewritten/**"],
    local_dir=tmpdir,
    token=TOKEN,
)

rewritten_dir = os.path.join(tmpdir, "rewritten")
assert os.path.isdir(rewritten_dir), "rewritten/ not found in download"

print(f"Uploading rewritten/ to {DST} ...")
api.upload_folder(
    folder_path=rewritten_dir,
    path_in_repo="rewritten",
    repo_id=DST,
    repo_type="dataset",
    commit_message="Add rewritten/ from crof.ai Layer 1 (12,375 EQ-Matrix rewritten samples)",
)
print(f"  ✓ rewritten/ uploaded to {DST}")

# ── Step 2: delete filtered/ from src ────────────────────────────────────────
print(f"\nListing filtered/ in {SRC} ...")
all_files    = list(api.list_repo_files(SRC, repo_type="dataset", token=TOKEN))
filtered     = [f for f in all_files if f.startswith("filtered/")]

if filtered:
    print(f"  Deleting {len(filtered)} file(s): {filtered}")
    api.create_commit(
        repo_id=SRC,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=f) for f in filtered],
        commit_message="Remove filtered/ (Layer 2 v1 results, 5.9% keep rate — will re-run)",
    )
    print(f"  ✓ filtered/ deleted from {SRC}")
else:
    print("  No filtered/ files found, skipping.")

print("\nDone. Summary:")
print(f"  {DST}/rewritten/  ← Layer 1 output (12,375 samples)")
print(f"  {SRC}/filtered/   ← deleted")
