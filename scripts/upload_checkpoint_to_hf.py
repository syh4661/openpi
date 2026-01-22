#!/usr/bin/env python3
"""
Upload OpenArm Pi0.5 checkpoint to Hugging Face Hub.

This script uploads the finetuned checkpoint to Hugging Face for easy sharing.
Large files are uploaded individually with progress tracking.

Usage:
    cd /home/saurabh/Development/openpi
    source .venv/bin/activate
    python scripts/upload_checkpoint_to_hf.py \
        --checkpoint-dir=./checkpoints/pi05_openarm/openarm_pi0.5_finetuned/5000 \
        --repo-name=openarm-pi05-finetuned \
        --private  # Optional: make repo private
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file
from tqdm import tqdm


def create_model_card(checkpoint_dir: Path, repo_id: str) -> str:
    """Create a README.md model card for the checkpoint."""
    
    model_card = f"""---
license: apache-2.0
tags:
  - robotics
  - manipulation
  - pi0
  - openpi
  - openarm
  - bimanual
  - isaac-sim
library_name: openpi
pipeline_tag: robotics
---

# OpenArm Pi0.5 Finetuned Checkpoint

This is a finetuned [Pi0.5](https://github.com/Physical-Intelligence/openpi) model checkpoint for the **OpenArm bimanual robot**.

## Model Details

- **Base Model**: Pi0.5 (Physical Intelligence)
- **Fine-tuned on**: OpenArm pick-and-place dataset
- **Action Dimension**: 16 (2 arms × (7 joints + 1 gripper))
- **Action Horizon**: 16 timesteps
- **Training Framework**: OpenPI (JAX/Flax)

## Usage

### Installation

```bash
git clone https://github.com/AiSaurabhPatil/openpi.git
cd openpi
uv sync
```

### Download & Inference

```python
from huggingface_hub import snapshot_download

# Download checkpoint
checkpoint_path = snapshot_download(
    repo_id="{repo_id}",
    local_dir="./checkpoints/pi05_openarm/openarm_pi0.5_finetuned/5000"
)

# Run inference server
# Terminal 1:
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 uv run scripts/serve_policy.py policy:checkpoint \\
#     --policy.config=pi05_openarm \\
#     --policy.dir=./checkpoints/pi05_openarm/openarm_pi0.5_finetuned/5000

# Terminal 2:
# python scripts/openarm_policy_client.py \\
#     --task="grab the electric screwdriver from the table and drop it inside the box"
```

### Isaac Sim Integration

This checkpoint is designed to work with Isaac Sim via ROS2:
1. Run the Isaac Sim OpenArm bridge
2. Start the policy server with this checkpoint
3. Run the policy client for inference

## Training Details

- **Dataset**: Custom OpenArm demonstrations recorded via teleoperation
- **Training Steps**: 5,000+
- **Batch Size**: 32
- **Learning Rate**: 2.5e-5 (cosine decay)
- **Hardware**: NVIDIA A100 80GB

## Files

```
├── params/           # Model weights (Orbax checkpoint format)
├── assets/           # Normalization statistics
└── _CHECKPOINT_METADATA  # Checkpoint metadata
```

## Citation

If you use this checkpoint, please cite:

```bibtex
@software{{openpi2025,
  author = {{Physical Intelligence}},
  title = {{OpenPI: Open-source Pi Foundation Models}},
  year = {{2025}},
  url = {{https://github.com/Physical-Intelligence/openpi}}
}}
```

## License

Apache 2.0
"""
    return model_card


def get_files_to_upload(checkpoint_dir: Path) -> list[tuple[Path, str, int]]:
    """Get list of files with their relative paths and sizes."""
    files = []
    for f in checkpoint_dir.rglob("*"):
        if f.is_file():
            rel_path = str(f.relative_to(checkpoint_dir))
            size = f.stat().st_size
            files.append((f, rel_path, size))
    # Sort by size (largest first) for better progress visibility
    files.sort(key=lambda x: x[2], reverse=True)
    return files


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/pi05_openarm/openarm_pi0.5_finetuned/5000",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="openarm-pi05-finetuned",
        help="Repository name on Hugging Face (will be created under your username)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Get HF username
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{args.repo_name}"
    
    # Get files to upload
    files = get_files_to_upload(checkpoint_dir)
    total_size = sum(f[2] for f in files)
    total_size_gb = total_size / (1024 ** 3)
    
    print(f"=" * 60)
    print(f"Hugging Face Checkpoint Upload")
    print(f"=" * 60)
    print(f"Username: {username}")
    print(f"Repository: {repo_id}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Private: {args.private}")
    print(f"Files: {len(files)}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print(f"=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN] Would upload the following files:")
        for f, rel_path, size in files:
            size_mb = size / (1024 * 1024)
            print(f"  {rel_path} ({size_mb:.1f} MB)")
        return
    
    # Create the repository
    print(f"\n📦 Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"✅ Repository created/exists: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️ Repository creation note: {e}")
    
    # Create and upload model card first
    print(f"\n📝 Creating model card...")
    model_card_content = create_model_card(checkpoint_dir, repo_id)
    readme_path = checkpoint_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card_content)
    print(f"✅ Model card created at: {readme_path}")
    
    # Refresh file list to include README
    files = get_files_to_upload(checkpoint_dir)
    total_size = sum(f[2] for f in files)
    
    # Upload files one by one with progress
    print(f"\n⬆️ Uploading {len(files)} files ({total_size_gb:.2f} GB)...")
    print(f"   Large files are uploaded individually for reliability.\n")
    
    uploaded_size = 0
    failed_files = []
    
    # Create overall progress bar
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Total progress") as pbar:
        for i, (file_path, rel_path, size) in enumerate(files):
            size_mb = size / (1024 * 1024)
            try:
                # Show which file we're uploading
                tqdm.write(f"[{i+1}/{len(files)}] Uploading: {rel_path} ({size_mb:.1f} MB)")
                
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload {rel_path}",
                )
                
                uploaded_size += size
                pbar.update(size)
                
            except Exception as e:
                tqdm.write(f"   ❌ Failed: {e}")
                failed_files.append((rel_path, str(e)))
    
    # Summary
    print(f"\n" + "=" * 60)
    if failed_files:
        print(f"⚠️ Upload completed with {len(failed_files)} failures:")
        for rel_path, error in failed_files:
            print(f"   ❌ {rel_path}: {error}")
    else:
        print(f"✅ All {len(files)} files uploaded successfully!")
    
    print(f"🔗 View your checkpoint: https://huggingface.co/{repo_id}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
