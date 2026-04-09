#!/usr/bin/env python3
"""
Upload LeRobot v2 dataset to Hugging Face Hub.

This script uploads a local LeRobot v2 dataset to Hugging Face Hub with:
- Proper dataset card with visualization support
- Progress tracking during upload
- Metadata for discoverability

Usage:
    cd /home/saurabh/Development/openpi
    source .venv/bin/activate
    python scripts/upload_dataset_to_hf.py \
        --dataset-path=~/.cache/huggingface/lerobot/saurabh/openarm_pick_v6 \
        --repo-name=openarm_pick_v6 \
        --private  # Optional: make repo private
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder
from tqdm import tqdm


def load_dataset_info(dataset_path: Path) -> dict:
    """Load dataset metadata from info.json."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info.json not found at: {info_path}")
    
    with open(info_path) as f:
        return json.load(f)


def load_tasks(dataset_path: Path) -> list[str]:
    """Load task descriptions from tasks.jsonl."""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    tasks = []
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    task_data = json.loads(line)
                    tasks.append(task_data.get("task", ""))
    return tasks


def create_dataset_card(dataset_path: Path, repo_id: str, info: dict, tasks: list[str]) -> str:
    """Create a README.md dataset card for the LeRobot dataset."""
    
    # Extract key info
    robot_type = info.get("robot_type", "unknown")
    total_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)
    total_videos = info.get("total_videos", 0)
    fps = info.get("fps", 30)
    codebase_version = info.get("codebase_version", "v2.0")
    
    # Get feature info
    features = info.get("features", {})
    state_shape = features.get("observation.state", {}).get("shape", [])
    action_shape = features.get("action", {}).get("shape", [])
    
    # Get camera names
    camera_keys = [k for k in features.keys() if k.startswith("observation.images.")]
    camera_names = [k.replace("observation.images.", "") for k in camera_keys]
    
    # Get video info from first camera
    video_info = {}
    if camera_keys:
        video_info = features[camera_keys[0]].get("info", {})
    
    # Task descriptions
    task_list = "\n".join([f"- {task}" for task in tasks]) if tasks else "- No task descriptions available"
    
    # Calculate approximate duration
    duration_seconds = total_frames / fps if fps > 0 else 0
    duration_minutes = duration_seconds / 60
    
    dataset_card = f"""---
license: apache-2.0
task_categories:
  - robotics
tags:
  - robotics
  - lerobot
  - manipulation
  - bimanual
  - isaac-sim
  - openarm
  - teleoperation
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/**/*.parquet"
---

# OpenArm Pick v6 - LeRobot Dataset

A LeRobot v2.1 dataset for bimanual robot manipulation, recorded in Isaac Sim with teleoperation.

## Dataset Description

This dataset contains teleoperated demonstrations of a bimanual OpenArm robot performing pick-and-place tasks. It was created for training the Pi0.5 model for robotic manipulation.

### Quick Stats

| Property | Value |
|----------|-------|
| **Robot** | {robot_type} |
| **Episodes** | {total_episodes} |
| **Total Frames** | {total_frames:,} |
| **Duration** | ~{duration_minutes:.1f} minutes |
| **FPS** | {fps} Hz |
| **Cameras** | {len(camera_names)} ({', '.join(camera_names)}) |
| **State Dimension** | {state_shape[0] if state_shape else 'N/A'} |
| **Action Dimension** | {action_shape[0] if action_shape else 'N/A'} |
| **Format** | LeRobot {codebase_version} |

### Tasks

{task_list}

### State/Action Space

The robot has 16 degrees of freedom (2 arms × (7 joints + 1 gripper)):

```
State/Action indices:
  [0-6]   Left arm joints (rad)
  [7]     Left gripper (normalized 0-1)
  [8-14]  Right arm joints (rad)
  [15]    Right gripper (normalized 0-1)
```

### Camera Configuration

| Camera | Resolution | Codec | FPS |
|--------|------------|-------|-----|
| head | {video_info.get('video.width', 480)}x{video_info.get('video.height', 360)} | {video_info.get('video.codec', 'av1')} | {video_info.get('video.fps', 30)} |
| wrist_left | {video_info.get('video.width', 480)}x{video_info.get('video.height', 360)} | {video_info.get('video.codec', 'av1')} | {video_info.get('video.fps', 30)} |
| wrist_right | {video_info.get('video.width', 480)}x{video_info.get('video.height', 360)} | {video_info.get('video.codec', 'av1')} | {video_info.get('video.fps', 30)} |

## Visualization

You can visualize this dataset using the LeRobot visualization space:

👉 **[Visualize Dataset](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset={repo_id})**

## Usage

### Loading with LeRobot

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("{repo_id}")

# Get a sample
sample = dataset[0]
print(sample.keys())
# dict_keys(['observation.state', 'action', 'observation.images.head', ...])
```

### Training with OpenPI

This dataset can be used directly with OpenPI for Pi0.5 finetuning:

```bash
cd openpi
uv run scripts/train.py pi05_openarm --exp-name=my_experiment
```

## Dataset Structure

```
{repo_id}/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.head/
│       ├── observation.images.wrist_left/
│       └── observation.images.wrist_right/
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── episodes_stats.jsonl
```

## Data Collection

- **Platform**: Isaac Sim (NVIDIA Omniverse)
- **Control**: VR teleoperation with Quest 3
- **Recording**: Custom ROS2 bridge + LeRobot recording script

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{openarm_pick_v6,
  author = {{AiSaurabhPatil}},
  title = {{OpenArm Pick v6 - LeRobot Dataset}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/datasets/{repo_id}}}
}}
```

## License

Apache 2.0
"""
    return dataset_card


def main():
    parser = argparse.ArgumentParser(description="Upload LeRobot dataset to Hugging Face Hub")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="~/.cache/huggingface/lerobot/saurabh/openarm_pick_v6",
        help="Path to local LeRobot dataset directory",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="openarm_pick_v6",
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
    
    # Expand path
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # Load dataset info
    print(f"📂 Loading dataset from: {dataset_path}")
    info = load_dataset_info(dataset_path)
    tasks = load_tasks(dataset_path)
    
    # Get HF username
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{args.repo_name}"
    
    print(f"\n{'=' * 60}")
    print(f"Hugging Face Dataset Upload")
    print(f"{'=' * 60}")
    print(f"Username: {username}")
    print(f"Repository: {repo_id}")
    print(f"Dataset: {dataset_path}")
    print(f"Private: {args.private}")
    print(f"\nDataset Info:")
    print(f"  Episodes: {info.get('total_episodes', 'N/A')}")
    print(f"  Frames: {info.get('total_frames', 'N/A'):,}")
    print(f"  FPS: {info.get('fps', 'N/A')}")
    print(f"  Robot: {info.get('robot_type', 'N/A')}")
    print(f"{'=' * 60}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would upload the following structure:")
        total_size = 0
        file_count = 0
        for f in dataset_path.rglob("*"):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                total_size += size_mb
                file_count += 1
                if file_count <= 20 or "meta/" in str(f):
                    print(f"  {f.relative_to(dataset_path)} ({size_mb:.2f} MB)")
        if file_count > 20:
            print(f"  ... and {file_count - 20} more files")
        print(f"\nTotal: {file_count} files, {total_size:.1f} MB")
        return
    
    # Create the repository
    print(f"\n📦 Creating dataset repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        print(f"✅ Repository created/exists: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"⚠️ Repository creation note: {e}")
    
    # Create and save dataset card
    print(f"\n📝 Creating dataset card...")
    dataset_card = create_dataset_card(dataset_path, repo_id, info, tasks)
    readme_path = dataset_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(dataset_card)
    print(f"✅ Dataset card created at: {readme_path}")
    
    # Upload the dataset folder with multi-commit for reliability
    print(f"\n⬆️  Uploading dataset (~1.1GB, this may take a while)...")
    print(f"   Using multi-commit upload for reliability...")
    
    try:
        upload_folder(
            folder_path=str(dataset_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload OpenArm LeRobot v2 dataset",
            multi_commits=True,  # Split into multiple commits for reliability
            multi_commits_verbose=True,  # Show progress for each commit
        )
        print(f"\n✅ Upload complete!")
        print(f"🔗 View your dataset: https://huggingface.co/datasets/{repo_id}")
        print(f"👁️  Visualize: https://huggingface.co/spaces/lerobot/visualize_dataset?dataset={repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
