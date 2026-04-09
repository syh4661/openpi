"""Compute normalization statistics for OpenArm dataset.

This script directly reads parquet files to bypass LeRobot dataset loading issues
with newer LeRobot v2.1 format and older HuggingFace datasets library.

Usage:
    python scripts/compute_openarm_norm_stats.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import tqdm

import openpi.shared.normalize as normalize


def main():
    # Dataset path
    dataset_path = Path.home() / ".cache/huggingface/lerobot/saurabh/openarm_pick_v6"
    data_path = dataset_path / "data" / "chunk-000"
    
    # Output path for norm stats
    assets_dir = Path("assets") / "pi05_openarm" / "saurabh" / "openarm_pick_v6"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Keys to compute stats for
    keys = ["state", "actions"]
    
    # Initialize running statistics
    stats = {key: normalize.RunningStats() for key in keys}
    
    # Get all parquet files
    parquet_files = sorted(data_path.glob("episode_*.parquet"))
    print(f"Found {len(parquet_files)} episode files")
    
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
