# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & common commands

This repo uses **uv** to manage a Python 3.11 environment driven by `uv.lock`. `GIT_LFS_SKIP_SMUDGE=1` is required when syncing because LeRobot is pulled in as a git dependency.

- Install: `GIT_LFS_SKIP_SMUDGE=1 uv sync` then `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`
- RLDS extras (DROID training): create a 3.11 venv, then `uv sync --group rlds`
- Lint / format: `uv run ruff check .` and `uv run ruff format .` (line length 120, isort with `force-single-line`, excludes `third_party/` and `src/openpi/models_pytorch/transformers_replace/`)
- Pre-commit: `pre-commit install` then `pre-commit run --all-files` (runs `uv-lock` + ruff)
- Full tests: `uv run pytest` — `pytest.ini_options.testpaths = ["src", "scripts", "packages"]`
- Single test file: `uv run pytest src/openpi/shared/normalize_test.py`
- Single test: `uv run pytest src/openpi/training/data_loader_test.py::test_name`
- Tests requiring GPU/network/checkpoints/manual setup are gated by `@pytest.mark.manual` — they are skipped by default; pass `-m manual` to run them.

### Training, norm stats, and serving

Configs are registered in `src/openpi/training/config.py` under `_CONFIGS` (see e.g. `pi05_openarm`, `pi0_libero`, `pi05_droid`). The config name is the entrypoint argument for every script.

- Norm stats (must run before any training of a new config): `uv run scripts/compute_norm_stats.py --config-name <config>`
- JAX training: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<run> [--overwrite|--resume]`
- PyTorch training (single GPU): `uv run scripts/train_pytorch.py <config> --exp_name <run>`; multi-GPU via `uv run torchrun --standalone --nnodes=1 --nproc_per_node=<N> scripts/train_pytorch.py <config> --exp_name <run>`
- Serve a checkpoint: `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint_dir>` (websocket on port 8000, see `src/openpi/serving/websocket_policy_server.py`)
- Convert JAX → PyTorch checkpoint: `uv run examples/convert_jax_model_to_pytorch.py --config_name <config> --checkpoint_dir <jax> --output_path <out>`. Inference auto-detects PyTorch by the presence of `model.safetensors` in the checkpoint dir (`policies/policy_config.py`).

`OPENPI_DATA_HOME` overrides the default `~/.cache/openpi` checkpoint cache; checkpoints live in `gs://openpi-assets`.

### PyTorch path requires a one-time patch

`uv sync` alone is not sufficient for PyTorch training/inference. After install you must overlay vendored transformers files:

```
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

This adds AdaRMS, fixes activation precision, and allows non-updating KV cache. Because uv hardlinks, this mutates the shared transformers cache — undo with `uv cache clean transformers`. Several PyTorch features remain unsupported: pi0-FAST, mixed-precision, FSDP, LoRA, and EMA.

## Architecture

`openpi` is a JAX-first vision-language-action (VLA) model repo with a parallel PyTorch implementation. Both share the same config, transform pipeline, and policy server contract.

### Layered request flow

A trained policy is a `Policy` (`src/openpi/policies/policy.py`) wrapping a model plus an ordered transform pipeline assembled in `policies/policy_config.create_trained_policy`:

```
raw runtime obs
  → repack_transforms.inputs                 (rename/reshape into the dataset’s schema)
  → InjectDefaultPrompt                      (fill prompt if caller omitted)
  → data_config.data_transforms.inputs       (robot-specific Inputs class, e.g. OpenArmInputs)
  → Normalize(norm_stats)                    (loaded from <ckpt>/assets/<asset_id>/)
  → data_config.model_transforms.inputs      (tokenize prompt, pad to action_dim, etc.)
  → model.sample_actions(...)
  → data_config.model_transforms.outputs
  → Unnormalize(norm_stats)
  → data_config.data_transforms.outputs      (e.g. OpenArmOutputs maps actions back to robot keys)
  → repack_transforms.outputs
```

When extending to a new robot, the work is almost always (a) a new `*_policy.py` `Inputs`/`Outputs` pair and (b) a new `TrainConfig` in `training/config.py` referencing it. Norm stats are loaded from the checkpoint, not the working tree, so policies in production carry the exact normalization used during training.

### Models

- `src/openpi/models/` is the JAX/Flax-NNX implementation: π₀ (`pi0.py`), π₀-FAST (`pi0_fast.py`), and shared backbones (`gemma.py`, `gemma_fast.py`, `siglip.py`, `vit.py`, `lora.py`). Configs live in `pi0_config.Pi0Config` (set `pi05=True` for π₀.₅; `action_dim`, `action_horizon`, `max_token_len` are commonly tuned per-robot).
- `src/openpi/models_pytorch/` mirrors the same architecture in PyTorch. The `transformers_replace/` subtree is **vendored transformers source** — it is excluded from ruff and from routine edits, and is overlaid into the live transformers package as described above.
- Training entrypoints `scripts/train.py` (JAX) and `scripts/train_pytorch.py` (PyTorch) both accept the same config names but feed different model objects. Sharding lives in `training/sharding.py`; both data loaders go through `training/data_loader.py` (LeRobot-backed) or `training/droid_rlds_dataset.py` (RLDS path).

### OpenArm runtime contract

`src/openpi/policies/openarm_runtime_contract.py` is the canonical source of truth for the bimanual OpenArm runtime — independent of any single runner:

- 16-D state/action ordering (`STATE_ORDER`, identical to `ACTION_ORDER`): left arm joints 1–7 + left gripper, then right arm joints 1–7 + right gripper.
- Camera keys are exactly `head`, `wrist_left`, `wrist_right`. `RUNTIME_CAMERA_NAME_TO_KEY` provides the only allowed aliasing (`chest → head`, `left_wrist → wrist_left`, etc.). `CAMERA_SERIALS` pins the physical RealSense devices.
- Image layout for runtime payloads is **CHW uint8**, shape `(3, 224, 224)`; fixture/test bundles use HWC `(224, 224, 3)`.
- Joints are degrees, grippers are normalized `0=open, 1=closed`, action chunk is 16 sequential **absolute** targets.
- Use `validate_runtime_observation`, `validate_action_chunk`, `validate_runtime_metadata`, and `create_runtime_policy` rather than re-implementing schema checks. `metadata()` returns the contract version (`openarm_runtime_contract/v1`).

`scripts/openarm_inference.py`, `scripts/openarm_policy_client.py`, `scripts/openarm_direct_runtime.py`, and `scripts/check_openarm_contract.py` are runners and validators that all go through this contract.

## Conventions

- Python 3.11 only — `requires-python = ">=3.11"` and tensorflow-cpu 2.15 (rlds group) only ships cp311 wheels.
- Tests live next to the module they cover and are named `*_test.py` (e.g. `model_test.py`, `policy_test.py`). Mark tests that need GPUs/credentials/large downloads with `@pytest.mark.manual`.
- `snake_case` everywhere except classes (`PascalCase`). Keep imports single-line (ruff isort `force-single-line`).
- Do not commit checkpoints, datasets, W&B credentials, or local cache paths.
- Do not edit `third_party/` or `src/openpi/models_pytorch/transformers_replace/` as part of normal work — both are excluded from ruff and represent vendored code.
- Commits use Conventional Commit style subjects (`feat:`, `fix:`, `docs:`); keep them imperative and one-change-per-PR.
