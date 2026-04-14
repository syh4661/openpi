"""Compute normalization statistics for OpenArm dataset.

This script directly reads parquet files to bypass LeRobot dataset loading issues
with newer LeRobot v2.1 format and older HuggingFace datasets library.

Usage:
    python scripts/compute_openarm_norm_stats.py
"""

from pathlib import Path

from huggingface_hub import snapshot_download
import numpy as np
import polars as pl
import tqdm

import openpi.shared.normalize as normalize


PRIMARY_REPO_ID = "saurabh/openarm_pick_v6"
FALLBACK_REPO_ID = "TetraxCode/openarm_pick_place_v1"
OUTPUT_REPO_ID = PRIMARY_REPO_ID


def _cache_path_for_repo(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def _resolve_dataset_path() -> tuple[str, Path]:
    primary_path = _cache_path_for_repo(PRIMARY_REPO_ID)
    if (primary_path / "data").exists():
        print(f"Using primary dataset cache: {PRIMARY_REPO_ID}")
        return PRIMARY_REPO_ID, primary_path

    fallback_path = _cache_path_for_repo(FALLBACK_REPO_ID)
    if not (fallback_path / "data").exists():
        print(f"Primary dataset unavailable, downloading fallback: {FALLBACK_REPO_ID}")
        fallback_path = Path(
            snapshot_download(
                repo_id=FALLBACK_REPO_ID,
                repo_type="dataset",
                local_dir=str(fallback_path),
                allow_patterns=["data/**", "meta/**"],
            )
        )

    print(f"Using fallback dataset source: {FALLBACK_REPO_ID}")
    return FALLBACK_REPO_ID, fallback_path


def _get_parquet_files(dataset_path: Path) -> list[Path]:
    data_path = dataset_path / "data"
    parquet_files = sorted(data_path.glob("chunk-*/episode_*.parquet"))
    if parquet_files:
        return parquet_files

    parquet_files = sorted(data_path.glob("chunk-*/file-*.parquet"))
    if parquet_files:
        return parquet_files

    raise FileNotFoundError(f"No parquet files found under {data_path}")


def main():
    source_repo_id, dataset_path = _resolve_dataset_path()

    # Output path for norm stats
    assets_dir = Path("assets") / "pi05_openarm" / OUTPUT_REPO_ID
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Keys to compute stats for
    keys = ["state", "actions"]

    # Initialize running statistics
    stats = {key: normalize.RunningStats() for key in keys}

    # Get all parquet files
    parquet_files = _get_parquet_files(dataset_path)
    print(f"Found {len(parquet_files)} parquet files from {source_repo_id}")

    # Process each episode
    for parquet_file in tqdm.tqdm(parquet_files, desc="Processing episodes"):
        df = pl.read_parquet(parquet_file)

        # observation.state -> state
        if "observation.state" in df.columns:
            state_data = np.stack(df["observation.state"].to_list())
            stats["state"].update(state_data)

        # action -> actions
        if "action" in df.columns:
            action_data = np.stack(df["action"].to_list())
            stats["actions"].update(action_data)

    # Compute final statistics
    norm_stats = {key: running_stats.get_statistics() for key, running_stats in stats.items()}

    # Print summary
    print("\n=== Normalization Statistics ===")
    for key, stat in norm_stats.items():
        print(f"\n{key}:")
        print(f"  Mean: {stat.mean}")
        print(f"  Std:  {stat.std}")
        print(f"  Q01:  {stat.q01}")
        print(f"  Q99:  {stat.q99}")

    # Save statistics
    output_path = assets_dir
    print(f"\nWriting stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print("Done!")


if __name__ == "__main__":
    main()
