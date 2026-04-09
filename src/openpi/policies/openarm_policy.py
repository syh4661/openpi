"""OpenArm bimanual robot policy transforms for OpenPI.

This module provides input and output transforms for the OpenArm bimanual robot platform.
The OpenArm robot has two 7-DOF arms with grippers, resulting in 16-dimensional state and action spaces.

State/Action Layout (16-dim):
- Indices 0-6: Left arm joints (joint1-7)
- Index 7: Left gripper
- Indices 8-14: Right arm joints (joint1-7)
- Index 15: Right gripper
"""

import dataclasses

import openpi.models.model as _model
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    """Transform inputs from OpenArm dataset format to model format.

    This transform handles:
    - Repacking images from LeRobot format to model format
    - Combining state vectors (16-dim: 2 arms × [7 joints + 1 gripper])

    Args:
        model_type: The model type (PI0, PI05, or PI0_FAST)
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        """Transform OpenArm data to model input format."""
        import einops
        import numpy as np

        def convert_image(img):
            """Convert image from LeRobot format to model format."""
            img = np.asarray(img)
            # Convert to uint8 if using float images
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel]
            return einops.rearrange(img, "c h w -> h w c")

        # Extract and convert images - OpenArm has 3 cameras
        # Map OpenArm camera names to model expected names:
        #   head -> base_0_rgb
        #   wrist_left -> left_wrist_0_rgb
        #   wrist_right -> right_wrist_0_rgb
        images = {}
        image_masks = {}
        if "head" in data:
            images["base_0_rgb"] = convert_image(data["head"])
            image_masks["base_0_rgb"] = np.True_
        if "wrist_left" in data:
            images["left_wrist_0_rgb"] = convert_image(data["wrist_left"])
            image_masks["left_wrist_0_rgb"] = np.True_
        if "wrist_right" in data:
            images["right_wrist_0_rgb"] = convert_image(data["wrist_right"])
            image_masks["right_wrist_0_rgb"] = np.True_

        # State is already in the correct format (16-dim)
        # Layout: [left_joint1-7, left_gripper, right_joint1-7, right_gripper]
        state = data.get("state")
        if state is not None:
            state = np.asarray(state)

        output = {"image": images, "image_mask": image_masks}
        if state is not None:
            output["state"] = state

        # Pass through other keys (e.g., prompt, actions)
        for key in ["prompt", "actions"]:
            if key in data:
                output[key] = data[key]

        return output


@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    """Transform model outputs to OpenArm action format.

    This is the inverse of OpenArmInputs, converting model predictions
    back to the robot's action space.
    """

    def __call__(self, data: dict) -> dict:
        """Transform model output to OpenArm action format."""
        # Actions are already in the correct format (16-dim)
        # No additional transformation needed for OpenArm
        return data
