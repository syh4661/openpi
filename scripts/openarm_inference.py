#!/usr/bin/env python
"""
OpenArm Pi0.5 Inference Service for Isaac Sim.

This script runs the finetuned Pi0.5 model and communicates with Isaac Sim via ROS2.
It uses the openpi policy loading infrastructure which handles all normalization
and transforms automatically.

Usage:
    # First start the policy server (in one terminal):
    cd /home/saurabh/Development/openpi
    source .venv/bin/activate
    source /opt/ros/humble/setup.bash
    python scripts/openarm_inference.py \
        --checkpoint-dir=./checkpoints/pi05_openarm/openarm_pi0.5_finetuned/5000 \
        --task="Pick up the drill machine from table and put it inside the box"

Note: Run the Isaac Sim OpenArm bridge in another terminal first.
"""

import argparse
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, JointState
except ImportError as e:
    logger.error("ROS2 not found. Run: source /opt/ros/humble/setup.bash")
    raise

# OpenPI imports
try:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
except ImportError as e:
    logger.error("OpenPI not found. Make sure you're in the openpi environment.")
    raise


@dataclass
class InferenceConfig:
    """Configuration for OpenArm Pi0.5 inference service."""
    
    # ROS2 topics (must match Isaac Sim bridge)
    camera_topics: dict = field(default_factory=lambda: {
        "head": "/camera/head/image_raw",
        "wrist_left": "/camera/wrist_left/image_raw",
        "wrist_right": "/camera/wrist_right/image_raw",
    })
    joint_state_topic: str = "/joint_states"
    joint_command_topic: str = "/joint_command"
    
    # Joint configuration (OpenArm bimanual: 16 training dims, but 18 actual joints)
    # The training uses 16-dim: [left_joints(7), left_gripper(1), right_joints(7), right_gripper(1)]
    # But actual Isaac Sim has 18 joints (2 gripper fingers per hand)
    joint_names: list = field(default_factory=lambda: [
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
        "openarm_left_joint7",
        "openarm_left_finger_joint1", "openarm_left_finger_joint2",  # Left gripper
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
        "openarm_right_joint7",
        "openarm_right_finger_joint1", "openarm_right_finger_joint2",  # Right gripper
    ])
    
    # Model training uses 16-dim state (single gripper value per hand)
    training_dim: int = 16
    
    # Checkpoint settings
    checkpoint_dir: str = "./checkpoints/pi05_openarm/openarm_pi0.5_finetuned/5000"
    config_name: str = "pi05_openarm"
    
    # Image settings (Pi0.5 expects 224x224)
    image_resolution: tuple = (224, 224)
    
    # Control settings
    fps: float = 10.0  # Inference frequency
    max_steps: int = 10000


class OpenArmInferenceNode(Node):
    """ROS2 node for OpenArm Pi0.5 inference."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__("openarm_pi05_inference")
        self.config = config
        
        # QoS for sensors
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        # Observation buffers (thread-safe)
        self._images = {name: None for name in config.camera_topics}
        self._joint_state = None
        self._obs_lock = threading.Lock()
        
        # Subscribe to cameras
        self._image_subs = []
        for cam_name, topic in config.camera_topics.items():
            sub = self.create_subscription(
                Image,
                topic,
                lambda msg, name=cam_name: self._image_callback(msg, name),
                sensor_qos,
            )
            self._image_subs.append(sub)
            self.get_logger().info(f"Subscribed to camera: {topic}")
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            config.joint_state_topic,
            self._joint_state_callback,
            sensor_qos,
        )
        self.get_logger().info(f"Subscribed to: {config.joint_state_topic}")
        
        # Publisher for actions
        self.action_pub = self.create_publisher(JointState, config.joint_command_topic, 10)
        self.get_logger().info(f"Publishing to: {config.joint_command_topic}")
    
    def _image_callback(self, msg: Image, cam_name: str):
        """Handle camera image."""
        try:
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = img[:, :, ::-1]  # BGR to RGB
            elif msg.encoding == 'rgba8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                img = img[:, :, :3]  # Drop alpha
            else:
                return
            
            with self._obs_lock:
                self._images[cam_name] = img.copy()
        except Exception as e:
            self.get_logger().warning(f"Image callback error: {e}")
    
    def _joint_state_callback(self, msg: JointState):
        """Handle joint state message from Isaac Sim.
        
        Converts 18-joint format to 16-dim training format:
        - Takes average of two gripper fingers for each hand
        """
        with self._obs_lock:
            # Create name-to-position map
            incoming_positions = dict(zip(msg.name, msg.position))
            
            # Build 16-dim state: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
            state_16 = np.zeros(self.config.training_dim, dtype=np.float32)
            
            # Left arm joints (0-6)
            for i in range(7):
                joint_name = f"openarm_left_joint{i+1}"
                if joint_name in incoming_positions:
                    state_16[i] = incoming_positions[joint_name]
            
            # Left gripper (7) - average of two fingers
            left_g1 = incoming_positions.get("openarm_left_finger_joint1", 0.0)
            left_g2 = incoming_positions.get("openarm_left_finger_joint2", 0.0)
            state_16[7] = (left_g1 + left_g2) / 2.0
            
            # Right arm joints (8-14)
            for i in range(7):
                joint_name = f"openarm_right_joint{i+1}"
                if joint_name in incoming_positions:
                    state_16[8 + i] = incoming_positions[joint_name]
            
            # Right gripper (15) - average of two fingers
            right_g1 = incoming_positions.get("openarm_right_finger_joint1", 0.0)
            right_g2 = incoming_positions.get("openarm_right_finger_joint2", 0.0)
            state_16[15] = (right_g1 + right_g2) / 2.0
            
            self._joint_state = state_16
    
    def get_observation(self) -> dict | None:
        """Get current observation in openpi format."""
        with self._obs_lock:
            # Check if all data is available
            if self._joint_state is None:
                return None
            if any(img is None for img in self._images.values()):
                return None
            
            # OpenPI expects images in HWC format (which is what we receive from ROS)
            # Also resize to 224x224 if needed
            images = {}
            for cam_name, img in self._images.items():
                if img.shape[:2] != self.config.image_resolution:
                    import cv2
                    img = cv2.resize(img, self.config.image_resolution)
                images[cam_name] = img.copy()
            
            return {
                # Map to dataset camera names that OpenArmInputs expects
                "head": images["head"],
                "wrist_left": images["wrist_left"],
                "wrist_right": images["wrist_right"],
                "state": self._joint_state.copy(),
            }
    
    def publish_action(self, action_16: np.ndarray):
        """Publish 16-dim action, expanding gripper to two fingers.
        
        Converts from training format (16-dim) to Isaac Sim format (18 joints):
        - Single gripper value is duplicated to both finger joints
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.config.joint_names
        
        # Build 18-dim action from 16-dim
        action_18 = np.zeros(18, dtype=np.float32)
        
        # Left arm (0-6) -> (0-6)
        action_18[0:7] = action_16[0:7]
        
        # Left gripper (7) -> (7, 8) - duplicate to both fingers
        action_18[7] = action_16[7]
        action_18[8] = action_16[7]
        
        # Right arm (8-14) -> (9-15)
        action_18[9:16] = action_16[8:15]
        
        # Right gripper (15) -> (16, 17) - duplicate to both fingers
        action_18[16] = action_16[15]
        action_18[17] = action_16[15]
        
        msg.position = action_18.tolist()
        self.action_pub.publish(msg)


def run_inference(config: InferenceConfig, task: str):
    """Main inference loop."""
    
    # Initialize ROS2
    rclpy.init()
    ros_node = OpenArmInferenceNode(config)
    
    # Spin ROS2 in background
    spin_thread = threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True)
    spin_thread.start()
    
    # Load Pi0.5 policy using openpi infrastructure
    logger.info(f"Loading Pi0.5 policy from: {config.checkpoint_dir}")
    logger.info(f"Using config: {config.config_name}")
    
    try:
        # Get the training config
        train_config = _config.get_config(config.config_name)
        
        # Create the policy with all transforms and normalization handled automatically
        policy = _policy_config.create_trained_policy(
            train_config,
            config.checkpoint_dir,
            default_prompt=task,  # Inject task as default prompt
        )
        
        logger.info("✅ Pi0.5 policy loaded successfully!")
        logger.info(f"Task: {task}")
        
    except Exception as e:
        logger.error(f"Failed to load policy: {e}")
        import traceback
        traceback.print_exc()
        ros_node.destroy_node()
        rclpy.shutdown()
        return
    
    # Wait for observations
    logger.info("Waiting for observations from Isaac Sim...")
    while ros_node.get_observation() is None:
        time.sleep(0.1)
    logger.info("✅ Receiving data from Isaac Sim!")
    
    # Inference loop
    logger.info("=" * 60)
    logger.info(f"Starting OpenArm Pi0.5 Inference")
    logger.info(f"Task: {task}")
    logger.info(f"FPS: {config.fps}")
    logger.info("=" * 60)
    
    frame_count = 0
    
    try:
        while frame_count < config.max_steps:
            loop_start = time.perf_counter()
            
            # Get observation
            obs = ros_node.get_observation()
            if obs is None:
                time.sleep(0.01)
                continue
            
            # Run inference
            # The policy.infer() method handles all transforms:
            # 1. OpenArmInputs: converts camera keys and image formats
            # 2. Normalize: applies quantile normalization from training stats
            # 3. Model inference: runs pi0.5 flow matching
            # 4. Unnormalize: converts actions back to original scale
            # 5. OpenArmOutputs: any final action processing
            try:
                result = policy.infer(obs)
                action = result["actions"]
                
                # Take first action from action horizon
                if action.ndim > 1:
                    action = action[0]
                
                # Ensure numpy array
                action_np = np.asarray(action, dtype=np.float32)
                
                # Publish to Isaac Sim
                ros_node.publish_action(action_np)
                
                # Debug logging
                if frame_count % 30 == 0:
                    logger.info(f"[Frame {frame_count}] Action range: [{action_np.min():.4f}, {action_np.max():.4f}]")
                
            except Exception as e:
                logger.warning(f"Inference error: {e}")
                import traceback
                traceback.print_exc()
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Frame {frame_count}")
            
            # Maintain FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = (1.0 / config.fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        logger.info("Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="OpenArm Pi0.5 Inference Service for Isaac Sim")
    parser.add_argument(
        "--task", type=str, required=True,
        help="Task description (e.g., 'Pick up the drill machine from table and put it inside the box')"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, 
        default="./checkpoints/pi05_openarm/openarm_pi0.5_finetuned/5000",
        help="Path to finetuned checkpoint directory"
    )
    parser.add_argument(
        "--config", type=str, default="pi05_openarm",
        help="Training config name"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000,
        help="Maximum number of inference steps"
    )
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="Inference frequency (frames per second)"
    )
    
    args = parser.parse_args()
    
    config = InferenceConfig(
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config,
        max_steps=args.max_steps,
        fps=args.fps,
    )
    
    run_inference(config, args.task)


if __name__ == "__main__":
    main()
