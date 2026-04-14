"""Canonical OpenArm runtime contract owned by OpenPI.

This module freezes the runtime-facing OpenArm policy schema independently from
the current local runner drift. The contract here is the single canonical source
for runtime observation keys, state/action ordering, image layout, units, and
action-chunk semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import functools
import hashlib
import importlib
import pathlib
import sys
import types

import numpy as np


class ContractError(ValueError):
    """Raised when an OpenArm runtime payload violates the canonical contract."""


METADATA_VERSION = "openarm_runtime_contract/v1"
RUNTIME_CONFIG_ALIAS = "pi05_openarm"
RUNTIME_CONFIG_NAME = "pi05_openarm_runtime"

STATE_ORDER = (
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_joint_7.pos",
    "left_gripper.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_joint_7.pos",
    "right_gripper.pos",
)
ACTION_ORDER = STATE_ORDER

CANONICAL_INPUT_KEYS = ("head", "wrist_left", "wrist_right", "state", "prompt")
CANONICAL_CAMERA_KEYS = ("head", "wrist_left", "wrist_right")
RUNTIME_CAMERA_NAME_TO_KEY = {
    "chest": "head",
    "left_wrist": "wrist_left",
    "right_wrist": "wrist_right",
    "wrist_a": "wrist_left",
    "wrist_b": "wrist_right",
}
CAMERA_SERIALS = {
    "head": "234322070493",
    "wrist_left": "230322273311",
    "wrist_right": "315122270766",
}

RUNTIME_IMAGE_SHAPE = (3, 224, 224)
FIXTURE_IMAGE_SHAPE = (224, 224, 3)
STATE_SHAPE = (16,)
ACTION_SHAPE = (16,)
ACTION_CHUNK_SHAPE = (16, 16)

JOINT_UNIT = "degrees"
GRIPPER_RANGE = (0.0, 1.0)
GRIPPER_SEMANTICS = "0=open, 1=closed"
IMAGE_LAYOUT = "CHW"
IMAGE_DTYPE = np.uint8
IMAGE_RANGE = (0, 255)
ACTION_SEMANTICS = "absolute"
ACTION_CHUNK_SEMANTICS = "16 sequential absolute targets in canonical state order"


@dataclasses.dataclass(frozen=True)
class ContractMetadata:
    version: str = METADATA_VERSION
    image_layout: str = IMAGE_LAYOUT
    image_dtype: str = np.dtype(IMAGE_DTYPE).name
    image_range: tuple[int, int] = IMAGE_RANGE
    state_shape: tuple[int, ...] = STATE_SHAPE
    action_shape: tuple[int, ...] = ACTION_SHAPE
    action_chunk_shape: tuple[int, ...] = ACTION_CHUNK_SHAPE
    joint_unit: str = JOINT_UNIT
    gripper_semantics: str = GRIPPER_SEMANTICS
    action_semantics: str = ACTION_SEMANTICS
    action_chunk_semantics: str = ACTION_CHUNK_SEMANTICS


def metadata() -> dict[str, object]:
    return dataclasses.asdict(ContractMetadata())


def _as_str(value: object, *, field_name: str) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ContractError(f"{field_name} must be a scalar string, got shape {value.shape}.")
        value = value.item()
    if not isinstance(value, str):
        raise ContractError(f"{field_name} must be a string, got {type(value).__name__}.")
    value = value.strip()
    if not value:
        raise ContractError(f"{field_name} is required and cannot be empty.")
    return value


def _as_exact_list(values: object) -> list[str]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ContractError(f"Expected a 1D metadata array, got shape {array.shape}.")
    return [str(item) for item in array.tolist()]


def canonical_camera_key(name: str) -> str:
    try:
        return RUNTIME_CAMERA_NAME_TO_KEY[name]
    except KeyError as exc:
        raise ContractError(f"Unknown camera name '{name}'.") from exc


def validate_prompt(prompt: object) -> str:
    return _as_str(prompt, field_name="prompt")


def validate_state(state: object) -> np.ndarray:
    array = np.asarray(state, dtype=np.float32)
    if array.shape != STATE_SHAPE:
        raise ContractError(f"state must have shape {STATE_SHAPE}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ContractError("state must contain only finite float values.")
    for index in (7, 15):
        if not (GRIPPER_RANGE[0] <= float(array[index]) <= GRIPPER_RANGE[1]):
            raise ContractError(
                f"state gripper index {index} must be normalized to {GRIPPER_RANGE}, got {array[index]:.6f}."
            )
    return array


def validate_runtime_image(image: object, *, key: str) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != IMAGE_DTYPE:
        raise ContractError(f"{key} must be {np.dtype(IMAGE_DTYPE).name}, got {array.dtype}.")
    if array.shape != RUNTIME_IMAGE_SHAPE:
        raise ContractError(f"{key} must use canonical {IMAGE_LAYOUT} layout {RUNTIME_IMAGE_SHAPE}; got {array.shape}.")
    if array.min() < IMAGE_RANGE[0] or array.max() > IMAGE_RANGE[1]:
        raise ContractError(f"{key} must stay in inclusive range {IMAGE_RANGE}, got [{array.min()}, {array.max()}].")
    return array


def validate_runtime_observation(observation: Mapping[str, object]) -> dict[str, object]:
    keys = set(observation.keys())
    slash_keys = sorted(key for key in keys if "/" in key)
    if slash_keys:
        raise ContractError(
            "Slash-style observation keys are non-canonical for OpenArm runtime payloads: " + ", ".join(slash_keys)
        )

    expected = set(CANONICAL_INPUT_KEYS)
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ContractError("Observation keys must exactly match canonical contract: " + "; ".join(details))

    validated = {
        "head": validate_runtime_image(observation["head"], key="head"),
        "wrist_left": validate_runtime_image(observation["wrist_left"], key="wrist_left"),
        "wrist_right": validate_runtime_image(observation["wrist_right"], key="wrist_right"),
        "state": validate_state(observation["state"]),
        "prompt": validate_prompt(observation["prompt"]),
    }
    return validated


def validate_action_vector(action: object) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape != ACTION_SHAPE:
        raise ContractError(f"action vector must have shape {ACTION_SHAPE}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ContractError("action vector must contain only finite float values.")
    for index in (7, 15):
        if not (GRIPPER_RANGE[0] <= float(array[index]) <= GRIPPER_RANGE[1]):
            raise ContractError(
                f"action gripper index {index} must be normalized to {GRIPPER_RANGE}, got {array[index]:.6f}."
            )
    return array


def validate_action_chunk(
    actions: object,
    *,
    action_semantics: str = ACTION_SEMANTICS,
    chunk_semantics: str = "full_horizon_sequence",
) -> np.ndarray:
    if action_semantics != ACTION_SEMANTICS:
        raise ContractError(f"OpenArm actions are canonicalized as {ACTION_SEMANTICS}, not {action_semantics}.")
    if chunk_semantics != "full_horizon_sequence":
        raise ContractError(
            f"OpenArm action chunks are canonicalized as the full 16-step horizon, not {chunk_semantics}."
        )

    array = np.asarray(actions, dtype=np.float32)
    if array.shape != ACTION_CHUNK_SHAPE:
        raise ContractError(f"action chunk must have shape {ACTION_CHUNK_SHAPE}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ContractError("action chunk must contain only finite float values.")
    for index in (7, 15):
        if np.any((array[:, index] < GRIPPER_RANGE[0]) | (array[:, index] > GRIPPER_RANGE[1])):
            raise ContractError(f"action chunk gripper index {index} must stay normalized to {GRIPPER_RANGE}.")
    return array


def validate_fixture_metadata(bundle: Mapping[str, object]) -> None:
    state_order = _as_exact_list(bundle["state_key_order"])
    if state_order != list(STATE_ORDER):
        raise ContractError(f"Fixture state order mismatch: {state_order}")

    action_order = _as_exact_list(bundle["action_key_order"])
    if action_order != list(ACTION_ORDER):
        raise ContractError(f"Fixture action order mismatch: {action_order}")

    camera_names = _as_exact_list(bundle["camera_names"])
    if camera_names != ["chest", "left_wrist", "right_wrist"]:
        raise ContractError(f"Fixture camera inventory mismatch: {camera_names}")

    camera_serials = _as_exact_list(bundle["camera_serials"])
    expected_serials = [CAMERA_SERIALS[canonical_camera_key(name)] for name in camera_names]
    if camera_serials != expected_serials:
        raise ContractError(f"Fixture camera serial mismatch: {camera_serials}")


def _validate_fixture_image(image: object, *, key: str) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != IMAGE_DTYPE:
        raise ContractError(f"Fixture image {key} must be {np.dtype(IMAGE_DTYPE).name}, got {array.dtype}.")
    if array.shape != FIXTURE_IMAGE_SHAPE:
        raise ContractError(f"Fixture image {key} must have HWC shape {FIXTURE_IMAGE_SHAPE}, got {array.shape}.")
    return array


def canonical_observation_from_fixture(bundle: Mapping[str, object]) -> dict[str, object]:
    validate_fixture_metadata(bundle)
    observation = {
        "head": np.transpose(
            _validate_fixture_image(bundle["observation_chest_image"], key="observation_chest_image"), (2, 0, 1)
        ).copy(),
        "wrist_left": np.transpose(
            _validate_fixture_image(bundle["observation_left_wrist_image"], key="observation_left_wrist_image"),
            (2, 0, 1),
        ).copy(),
        "wrist_right": np.transpose(
            _validate_fixture_image(bundle["observation_right_wrist_image"], key="observation_right_wrist_image"),
            (2, 0, 1),
        ).copy(),
        "state": bundle["observation_state"],
        "prompt": bundle["prompt"],
    }
    return validate_runtime_observation(observation)


def validate_fixture_bundle(bundle: Mapping[str, object]) -> dict[str, object]:
    observation = canonical_observation_from_fixture(bundle)
    action_chunk = validate_action_chunk(bundle["action_chunk"])
    action_vector = validate_action_vector(bundle["action_vector"])
    if not np.allclose(action_chunk[0], action_vector):
        raise ContractError("Canonical fixture requires action_vector == action_chunk[0].")

    return {
        "metadata": metadata(),
        "observation": observation,
        "action_vector": action_vector,
        "action_chunk": action_chunk,
    }


def _config_source_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "training" / "config.py"


@functools.lru_cache(maxsize=1)
def _load_training_config_module_allowing_duplicates():
    config_path = _config_source_path()
    source = config_path.read_text()
    duplicate_guard = 'if len({config.name for config in _CONFIGS}) != len(_CONFIGS):\n    raise ValueError("Config names must be unique.")\n'
    if duplicate_guard not in source:
        raise RuntimeError(f"Could not locate duplicate-config guard in {config_path}.")
    source = source.replace(duplicate_guard, 'if False:\n    raise ValueError("Config names must be unique.")\n', 1)

    module_name = "openpi.training.config"
    module = types.ModuleType(module_name)
    module.__file__ = str(config_path)
    module.__package__ = "openpi.training"
    sys.modules[module_name] = module
    exec(compile(source, str(config_path), "exec"), module.__dict__)
    return module


@functools.lru_cache(maxsize=1)
def _load_training_config_module():
    try:
        return importlib.import_module("openpi.training.config")
    except ValueError as exc:
        if "Config names must be unique." not in str(exc):
            raise
        return _load_training_config_module_allowing_duplicates()


def load_openarm_train_config(config_name: str):
    config_module = _load_training_config_module()
    if config_name == RUNTIME_CONFIG_ALIAS:
        return config_module.get_config(RUNTIME_CONFIG_NAME)

    matches = [config for config in config_module._CONFIGS if config.name == config_name]
    if not matches:
        return config_module.get_config(config_name)
    if len(matches) == 1:
        return matches[0]

    for candidate in matches:
        base_config = getattr(getattr(candidate.data, "base_config", None), "prompt_from_task", None)
        if base_config is False:
            return candidate
    return matches[0]


def _resolve_checkpoint_dir(checkpoint_dir: pathlib.Path | str) -> pathlib.Path:
    download = importlib.import_module("openpi.shared.download")
    return pathlib.Path(download.maybe_download(str(checkpoint_dir))).resolve()


def _checkpoint_fingerprint(checkpoint_dir: pathlib.Path | str) -> tuple[pathlib.Path, str, str]:
    resolved = _resolve_checkpoint_dir(checkpoint_dir)
    if (resolved / "model.safetensors").exists():
        checkpoint_format = "pytorch"
    elif (resolved / "params").exists():
        checkpoint_format = "jax"
    else:
        checkpoint_format = "unknown"

    digest = hashlib.sha256(f"{resolved}|{checkpoint_format}".encode("utf-8")).hexdigest()[:16]
    return resolved, checkpoint_format, digest


def runtime_metadata(
    *,
    config_name: str,
    checkpoint_dir: pathlib.Path | str,
    policy_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_checkpoint, checkpoint_format, checkpoint_fingerprint = _checkpoint_fingerprint(checkpoint_dir)
    merged = metadata()
    merged.update(
        {
            "config_name": config_name,
            "canonical_input_keys": list(CANONICAL_INPUT_KEYS),
            "checkpoint_dir": str(resolved_checkpoint),
            "checkpoint_format": checkpoint_format,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "prompt_required": True,
            "state_order": list(STATE_ORDER),
            "action_order": list(ACTION_ORDER),
        }
    )
    if policy_metadata:
        merged["policy_metadata"] = dict(policy_metadata)
    return merged


def validate_runtime_metadata(metadata_dict: Mapping[str, object]) -> dict[str, object]:
    expected = metadata()
    validated = dict(metadata_dict)

    for key, expected_value in expected.items():
        if key not in validated:
            raise ContractError(f"Runtime metadata is missing required field '{key}'.")
        if validated[key] != expected_value:
            raise ContractError(f"Runtime metadata field '{key}' must be {expected_value!r}, got {validated[key]!r}.")

    if not isinstance(validated.get("config_name"), str) or not validated["config_name"]:
        raise ContractError("Runtime metadata must include a non-empty string config_name.")

    if _as_exact_list(validated.get("canonical_input_keys", ())) != list(CANONICAL_INPUT_KEYS):
        raise ContractError(f"Runtime metadata canonical_input_keys must be {list(CANONICAL_INPUT_KEYS)}.")

    if not isinstance(validated.get("checkpoint_dir"), str) or not validated["checkpoint_dir"]:
        raise ContractError("Runtime metadata must include a non-empty string checkpoint_dir.")

    if not isinstance(validated.get("checkpoint_format"), str) or not validated["checkpoint_format"]:
        raise ContractError("Runtime metadata must include a non-empty string checkpoint_format.")

    if not isinstance(validated.get("checkpoint_fingerprint"), str) or not validated["checkpoint_fingerprint"]:
        raise ContractError("Runtime metadata must include a non-empty string checkpoint_fingerprint.")

    if validated.get("prompt_required") is not True:
        raise ContractError("Runtime metadata prompt_required must be True.")

    if _as_exact_list(validated.get("state_order", ())) != list(STATE_ORDER):
        raise ContractError(f"Runtime metadata state_order must be {list(STATE_ORDER)}.")

    if _as_exact_list(validated.get("action_order", ())) != list(ACTION_ORDER):
        raise ContractError(f"Runtime metadata action_order must be {list(ACTION_ORDER)}.")

    policy_metadata = validated.get("policy_metadata")
    if policy_metadata is not None and not isinstance(policy_metadata, Mapping):
        raise ContractError("Runtime metadata policy_metadata must be a mapping when present.")

    return validated


def validate_policy_output(result: Mapping[str, object]) -> dict[str, object]:
    if "actions" not in result:
        raise ContractError("Policy result must include an 'actions' entry.")

    validated = dict(result)
    validated["actions"] = validate_action_chunk(result["actions"])
    return validated


class OpenArmRuntimePolicy:
    def __init__(self, policy, *, config_name: str, checkpoint_dir: pathlib.Path | str):
        self._policy = policy
        self._metadata = validate_runtime_metadata(
            runtime_metadata(
                config_name=config_name,
                checkpoint_dir=checkpoint_dir,
                policy_metadata=getattr(policy, "metadata", None),
            )
        )

    def infer(self, observation: Mapping[str, object]) -> dict[str, object]:
        canonical_observation = validate_runtime_observation(observation)
        result = validate_policy_output(self._policy.infer(canonical_observation))
        result["metadata"] = dict(self._metadata)
        return result

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)


def create_runtime_policy(
    *,
    config_name: str,
    checkpoint_dir: pathlib.Path | str,
    default_prompt: str | None = None,
):
    train_config = load_openarm_train_config(config_name)
    policy_config = importlib.import_module("openpi.policies.policy_config")
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir, default_prompt=default_prompt)
    return OpenArmRuntimePolicy(policy, config_name=config_name, checkpoint_dir=checkpoint_dir)
