#!/usr/bin/env python3
"""Run canonical OpenArm direct inference against a frozen fixture."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def _ensure_src_on_path() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return repo_root


def main() -> int:
    _ensure_src_on_path()

    from openpi.policies import openarm_runtime_contract as contract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory to load")
    parser.add_argument("--config", required=True, help="Training config name")
    parser.add_argument("--fixture", required=True, help="Path to canonical OpenArm fixture .npz")
    parser.add_argument("--output", required=True, help="Path to write runtime evidence .npz")
    parser.add_argument("--default-prompt", default=None, help="Optional prompt override if the observation omits one")
    args = parser.parse_args()

    fixture_path = pathlib.Path(args.fixture)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(fixture_path, allow_pickle=True) as fixture:
        bundle = {name: fixture[name] for name in fixture.files}

    print(f"[run] validating fixture: {fixture_path}", flush=True)
    validated = contract.validate_fixture_bundle(bundle)
    print(f"[run] loading policy: config={args.config} checkpoint={args.checkpoint_dir}", flush=True)
    runtime_policy = contract.create_runtime_policy(
        config_name=args.config,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    print("[run] running direct inference", flush=True)
    result = runtime_policy.infer(validated["observation"])

    np.savez_compressed(
        output_path,
        actions=np.asarray(result["actions"], dtype=np.float32),
        action_vector=np.asarray(result["actions"][0], dtype=np.float32),
        metadata_json=np.array(json.dumps(result["metadata"], sort_keys=True)),
        prompt=np.array(validated["observation"]["prompt"]),
    )

    print(f"[ok] fixture: {fixture_path}")
    print(f"[ok] output: {output_path}")
    print(f"[ok] actions.shape: {np.asarray(result['actions']).shape}")
    print(f"[ok] metadata.version: {result['metadata']['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
