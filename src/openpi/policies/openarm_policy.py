"""OpenArm bimanual policy transforms for Pi0.5."""
import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

def make_openarm_example() -> dict:
    return {
        "observation/chest_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(16).astype(np.float32),
        "prompt": "pick up the object",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        chest = _parse_image(data["observation/chest_image"])
        left_wrist = _parse_image(data["observation/left_wrist_image"])
        right_wrist = _parse_image(data["observation/right_wrist_image"])
        
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (chest, left_wrist, right_wrist)
        image_masks = (np.True_, np.True_, np.True_)
        
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes): prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs

@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {"actions": actions[:, :16]}
