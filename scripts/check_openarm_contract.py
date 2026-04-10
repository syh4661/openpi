#!/usr/bin/env python3
"""Validate the canonical OpenArm runtime contract against a frozen fixture."""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

import numpy as np


def _load_contract(module_name: str):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return importlib.import_module(module_name)


def _expect_rejection(label: str, fn, expected_substring: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - explicit contract failure reporting
        message = str(exc)
        if expected_substring not in message:
            raise AssertionError(
                f"{label} rejected for the wrong reason. Expected '{expected_substring}' in '{message}'."
            ) from exc
        print(f"[reject-ok] {label}: {message}")
        return
    raise AssertionError(f"{label} unexpectedly passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, help="Path to baseline_fixture.npz")
    parser.add_argument("--contract-module", required=True, help="Import path for the contract module")
    args = parser.parse_args()

    contract = _load_contract(args.contract_module)
    fixture_path = pathlib.Path(args.fixture)
    with np.load(fixture_path, allow_pickle=True) as fixture:
        bundle = {name: fixture[name] for name in fixture.files}

    validated = contract.validate_fixture_bundle(bundle)
    print(f"[ok] fixture: {fixture_path}")
    print(f"[ok] metadata_version: {validated['metadata']['version']}")
    print(f"[ok] canonical_keys: {contract.CANONICAL_INPUT_KEYS}")
    print(f"[ok] action_chunk_shape: {validated['action_chunk'].shape}")

    canonical_obs = validated["observation"]
    slash_variant = {
        "observation/chest_image": np.transpose(canonical_obs["head"], (1, 2, 0)),
        "observation/left_wrist_image": np.transpose(canonical_obs["wrist_left"], (1, 2, 0)),
        "observation/right_wrist_image": np.transpose(canonical_obs["wrist_right"], (1, 2, 0)),
        "observation/state": canonical_obs["state"],
        "prompt": canonical_obs["prompt"],
    }
    _expect_rejection(
        "legacy slash-style observation",
        lambda: contract.validate_runtime_observation(slash_variant),
        "Slash-style observation keys",
    )

    hwc_variant = {
        **canonical_obs,
        "head": np.transpose(canonical_obs["head"], (1, 2, 0)),
        "wrist_left": np.transpose(canonical_obs["wrist_left"], (1, 2, 0)),
        "wrist_right": np.transpose(canonical_obs["wrist_right"], (1, 2, 0)),
    }
    _expect_rejection(
        "legacy HWC image layout",
        lambda: contract.validate_runtime_observation(hwc_variant),
        "canonical CHW layout",
    )

    _expect_rejection(
        "delta step0-only action interpretation",
        lambda: contract.validate_action_chunk(
            validated["action_chunk"],
            action_semantics="delta",
            chunk_semantics="step0_only",
        ),
        "canonicalized as absolute",
    )

    print("[ok] legacy slash-key and non-canonical image/action variants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
