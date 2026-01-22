import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class FlexibleCheckpointWeightLoader(WeightLoader):
    """Loads weights from a checkpoint, skipping layers with shape mismatches.

    This is useful when fine-tuning with a different action dimension than the base model.
    Layers with mismatched shapes will use random initialization from the model params.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Regex pattern for keys to skip if shapes don't match (uses random init instead)
    skip_mismatched_regex: str = ".*action_(in|out)_proj.*"

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Merge with shape-aware logic
        return _merge_params_flexible(
            loaded_params, params, 
            missing_regex=".*lora.*",
            skip_mismatched_regex=self.skip_mismatched_regex
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params_flexible(
    loaded_params: at.Params, 
    params: at.Params, 
    *, 
    missing_regex: str,
    skip_mismatched_regex: str
) -> at.Params:
    """Merges loaded parameters with reference, skipping mismatched shapes for specified keys.

    This is useful when fine-tuning with a different action dimension than the base model.
    Layers matching skip_mismatched_regex with shape mismatches will use random init.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for missing keys to merge from reference.
        skip_mismatched_regex: A regex pattern for keys where shape mismatches should use reference values.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    skip_pattern = re.compile(skip_mismatched_regex)

    result = {}
    skipped_keys = []
    
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref_shape = flat_ref[k].shape
            loaded_shape = v.shape
            
            if ref_shape != loaded_shape:
                # Shape mismatch - check if we should skip this layer
                if skip_pattern.fullmatch(k):
                    # Use reference (random init) for this layer
                    logger.info(f"Shape mismatch at {k}: expected {ref_shape}, got {loaded_shape}. Using random init.")
                    skipped_keys.append(k)
                    result[k] = flat_ref[k]
                else:
                    # Shape mismatch on non-skippable layer - this is an error
                    raise ValueError(f"Shape mismatch at {k}: expected {ref_shape}, got {loaded_shape}")
            else:
                # Shapes match - use loaded weights
                result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    if skipped_keys:
        logger.info(f"Skipped {len(skipped_keys)} layers due to shape mismatch: {skipped_keys}")

    # Merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")

