# Repository Guidelines

## Project Structure & Module Organization

`src/openpi` contains the main Python package. Core model code lives in `src/openpi/models`, PyTorch-specific model code in `src/openpi/models_pytorch`, policy adapters in `src/openpi/policies`, training utilities in `src/openpi/training`, serving code in `src/openpi/serving`, and shared helpers in `src/openpi/shared`. Standalone workflows and entrypoints are in `scripts/`, including training, checkpoint upload, normalization statistics, and OpenArm runtime helpers. Robot and benchmark examples are under `examples/`; supporting documentation is in `docs/`. The `packages/` workspace currently includes `packages/openpi-client`.

## Build, Test, and Development Commands

- `GIT_LFS_SKIP_SMUDGE=1 uv sync`: install the Python 3.11 environment from `uv.lock`.
- `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`: install the repository in editable mode.
- `uv run pytest`: run the test suite configured for `src`, `scripts`, and `packages`.
- `uv run pytest src/openpi/shared/normalize_test.py`: run a focused test file.
- `uv run ruff check .` and `uv run ruff format .`: lint and format Python code.
- `pre-commit install` then `pre-commit run --all-files`: run the same Ruff and lockfile checks used before PRs.

## Coding Style & Naming Conventions

Use Python 3.11 for the main package. Follow Ruff formatting with a 120-character line length. Imports are sorted by Ruff/isort with single-line imports preferred. Use `snake_case` for functions, variables, modules, config names, and test files; use `PascalCase` for classes. Keep generated or vendored code out of routine edits, especially `third_party/` and `src/openpi/models_pytorch/transformers_replace/`, which are excluded from Ruff.

## Testing Guidelines

Tests use `pytest`. Place tests beside related modules and name files `*_test.py`, as in `src/openpi/models/model_test.py` or `src/openpi/training/data_loader_test.py`. Mark tests that require hardware, credentials, large downloads, or manual setup with `@pytest.mark.manual`. Prefer focused unit tests for transforms, configs, data loaders, and policy behavior; run a targeted pytest command before the full suite when working on GPU- or checkpoint-heavy paths.

## Commit & Pull Request Guidelines

Recent history uses short Conventional Commit-style subjects such as `feat: add pi05_openarm config and policy transforms` and `docs: clarify Python 3.11 requirement for RLDS group`. Keep subjects imperative and scoped to one change. PRs should include a clear title, a concise description, linked issue or discussion when relevant, and the exact tests or manual robot/runtime checks performed. Include screenshots or logs only when they clarify UI, notebook, Docker, or runtime behavior.

## Security & Configuration Tips

Do not commit checkpoints, datasets, secrets, W&B credentials, or local cache paths. Checkpoints are normally downloaded from `gs://openpi-assets` and cached under `~/.cache/openpi`; override with `OPENPI_DATA_HOME` when needed. For RLDS dependencies, use Python 3.11 before running `uv sync --group rlds`.
