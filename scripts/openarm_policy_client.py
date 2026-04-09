#!/usr/bin/env python3
"""
OpenArm ROS2 Policy Client for Pi0.5 Inference.

This script runs with system Python 3.10 (ROS2 compatible) and:
1. Subscribes to ROS2 camera and joint state topics from Isaac Sim
2. Connects to the OpenPI policy server via websocket
3. Sends observations, receives actions
4. Publishes actions back to ROS2

Usage:
    # Terminal 1: Start policy server (OpenPI environment)
    cd /model_checkpoint_storage/openpi
    source .venv/bin/activate
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_openarm \
        --policy.dir=./checkpoints/pi05_openarm/openarm_pi0.5_finetuned/5000 \
        --default-prompt="Pick up the drill machine"

    # Terminal 2: Start Isaac Sim bridge
    ~/.local/share/ov/pkg/isaac-sim-*/python.sh isaac_openarm_ros_bridge.py

    # Terminal 3: Run this client (system Python with ROS2)
    source /opt/ros/humble/setup.bash
    python3 openarm_policy_client.py --host=localhost --port=8000 --fps=30.0
"""

import argparse
import logging
import threading
import time
from dataclasses import dataclass, field

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

# Websocket client - minimal dependencies
try:
    import websockets.sync.client
    import msgpack
except ImportError:
    logger.error("Missing dependencies. Install: pip install websockets msgpack")
    raise


@dataclass
class ClientConfig:
    """Configuration for OpenArm policy client."""
    
    # Policy server connection
    host: str = "localhost"
    port: int = 8000
    
    # ROS2 topics (must match Isaac Sim bridge)
    camera_topics: dict = field(default_factory=lambda: {
        "head": "/camera/head/image_raw",
        "wrist_left": "/camera/wrist_left/image_raw",
        "wrist_right": "/camera/wrist_right/image_raw",
    })
    joint_state_topic: str = "/joint_states"
    joint_command_topic: str = "/joint_command"
    
    # Joint configuration (18-joint Isaac Sim to 16-dim training)
    joint_names_18: list = field(default_factory=lambda: [
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
        "openarm_left_joint7",
        "openarm_left_finger_joint1", "openarm_left_finger_joint2",
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
        "openarm_right_joint7",
        "openarm_right_finger_joint1", "openarm_right_finger_joint2",
    ])
    
    # Gripper range configuration (Isaac Sim raw values)
    # OpenPI expects normalized [0, 1] where 0=open, 1=closed
    gripper_raw_open: float = 0.11    # Raw value when gripper is fully open
    gripper_raw_closed: float = 0.0   # Raw value when gripper is fully closed
    
    training_dim: int = 16
    image_resolution: tuple = (224, 224)
    fps: float = 30.0
    max_steps: int = 10000


def gripper_raw_to_normalized(raw_value: float, config: ClientConfig) -> float:
    """Convert Isaac Sim raw gripper value to OpenPI [0, 1]. 0=open, 1=closed."""
    raw_open = config.gripper_raw_open
    raw_closed = config.gripper_raw_closed
    if abs(raw_open - raw_closed) < 1e-6:
        return 0.5
    normalized = (raw_open - raw_value) / (raw_open - raw_closed)
    return float(np.clip(normalized, 0.0, 1.0))


def gripper_normalized_to_raw(normalized_value: float, config: ClientConfig) -> float:
    """Convert OpenPI [0, 1] to Isaac Sim raw gripper value."""
    raw_open = config.gripper_raw_open
    raw_closed = config.gripper_raw_closed
    raw_value = raw_open - normalized_value * (raw_open - raw_closed)
    return float(np.clip(raw_value, min(raw_open, raw_closed), max(raw_open, raw_closed)))


def pack_numpy(obj):
    """Pack numpy arrays for msgpack."""
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": str(obj.dtype),
            b"shape": obj.shape,
        }
    elif isinstance(obj, dict):
        return {k: pack_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [pack_numpy(v) for v in obj]
    return obj


def unpack_numpy(obj):
    """Unpack numpy arrays from msgpack."""
    if isinstance(obj, dict):
        if b"__ndarray__" in obj or "__ndarray__" in obj:
            data = obj.get(b"data", obj.get("data"))
            dtype = obj.get(b"dtype", obj.get("dtype"))
            shape = obj.get(b"shape", obj.get("shape"))
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        return {k: unpack_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [unpack_numpy(v) for v in obj]
    return obj


class PolicyClient:
    """Websocket client to connect to OpenPI policy server."""
    
    def __init__(self, host: str, port: int, action_horizon: int = 16):
        self.uri = f"ws://{host}:{port}"
        self.ws = None
        self.metadata = None
        self.action_horizon = action_horizon
        self._connect()
        
        # Action chunk broker state
        self._action_chunk = None  # Cached action chunk (action_horizon, action_dim)
        self._chunk_step = 0       # Current step within the chunk
    
    def _connect(self):
        """Connect to policy server."""
        logger.info(f"Connecting to policy server at {self.uri}...")
        while True:
            try:
                self.ws = websockets.sync.client.connect(
                    self.uri, compression=None, max_size=None
                )
                # Receive server metadata
                self.metadata = unpack_numpy(msgpack.unpackb(self.ws.recv()))
                logger.info(f"✅ Connected! Server metadata: {self.metadata}")
                return
            except ConnectionRefusedError:
                logger.info("Waiting for policy server...")
                time.sleep(2)
    
    def _raw_infer(self, observation: dict) -> dict:
        """Send observation and receive action chunk from server."""
        packed = msgpack.packb(pack_numpy(observation))
        self.ws.send(packed)
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error: {response}")
        return unpack_numpy(msgpack.unpackb(response))
    
    def infer(self, observation: dict, use_chunking: bool = True) -> tuple:
        """Get next action, using action chunking to reduce inference calls.
        
        Returns:
            tuple: (action, is_new_chunk, chunk_step)
        """
        if not use_chunking:
            # Original behavior - call inference every time
            result = self._raw_infer(observation)
            action = result["actions"]
            if hasattr(action, 'ndim') and action.ndim > 1:
                action = action[0]
            return np.asarray(action, dtype=np.float32), True, 0
        
        # Action chunking - reuse cached actions
        if self._action_chunk is None:
            # Need new chunk from server
            result = self._raw_infer(observation)
            self._action_chunk = np.asarray(result["actions"], dtype=np.float32)
            self._chunk_step = 0
            is_new_chunk = True
        else:
            is_new_chunk = False
        
        # Get action for current step
        action = self._action_chunk[self._chunk_step]
        current_step = self._chunk_step
        
        # Advance to next step
        self._chunk_step += 1
        if self._chunk_step >= self.action_horizon:
            self._action_chunk = None  # Request new chunk next time
        
        return action, is_new_chunk, current_step
    
    def reset(self):
        """Reset action chunk state (call when starting new episode)."""
        self._action_chunk = None
        self._chunk_step = 0


class OpenArmPolicyClientNode(Node):
    """ROS2 node that bridges Isaac Sim topics to policy server."""
    
    def __init__(self, config: ClientConfig):
        super().__init__("openarm_policy_client")
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
            
            # Resize to 224x224 if needed
            if img.shape[:2] != self.config.image_resolution:
                import cv2
                img = cv2.resize(img, self.config.image_resolution)
            
            with self._obs_lock:
                self._images[cam_name] = img.copy()
        except Exception as e:
            self.get_logger().warning(f"Image callback error: {e}")
    
    def _joint_state_callback(self, msg: JointState):
        """Handle joint state, converting 18-joint to 16-dim format."""
        with self._obs_lock:
            incoming_positions = dict(zip(msg.name, msg.position))
            
            # Build 16-dim state: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
            state_16 = np.zeros(self.config.training_dim, dtype=np.float32)
            
            # Left arm joints (0-6)
            for i in range(7):
                joint_name = f"openarm_left_joint{i+1}"
                if joint_name in incoming_positions:
                    state_16[i] = incoming_positions[joint_name]
            
            # Left gripper (7) - average of two fingers, then NORMALIZE for model
            left_g1 = incoming_positions.get("openarm_left_finger_joint1", 0.0)
            left_g2 = incoming_positions.get("openarm_left_finger_joint2", 0.0)
            left_gripper_raw = (left_g1 + left_g2) / 2.0
            state_16[7] = gripper_raw_to_normalized(left_gripper_raw, self.config)
            
            # Right arm joints (8-14)
            for i in range(7):
                joint_name = f"openarm_right_joint{i+1}"
                if joint_name in incoming_positions:
                    state_16[8 + i] = incoming_positions[joint_name]
            
            # Right gripper (15) - average of two fingers, then NORMALIZE for model
            right_g1 = incoming_positions.get("openarm_right_finger_joint1", 0.0)
            right_g2 = incoming_positions.get("openarm_right_finger_joint2", 0.0)
            right_gripper_raw = (right_g1 + right_g2) / 2.0
            state_16[15] = gripper_raw_to_normalized(right_gripper_raw, self.config)
            
            self._joint_state = state_16
    
    def get_observation(self) -> dict | None:
        """Get current observation for policy server."""
        with self._obs_lock:
            if self._joint_state is None:
                return None
            if any(img is None for img in self._images.values()):
                return None
            
            # Format for OpenArmInputs transform
            # OpenArmInputs expects images in CHW format (channel, height, width)
            # because it uses einops.rearrange(img, "c h w -> h w c")
            # So we need to transpose from HWC to CHW
            return {
                "head": np.transpose(self._images["head"].copy(), (2, 0, 1)),
                "wrist_left": np.transpose(self._images["wrist_left"].copy(), (2, 0, 1)),
                "wrist_right": np.transpose(self._images["wrist_right"].copy(), (2, 0, 1)),
                "state": self._joint_state.copy(),
            }
    
    def publish_action(self, action_16: np.ndarray):
        """Publish 16-dim action, expanding gripper to two fingers."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.config.joint_names_18
        
        # Build 18-dim action from 16-dim
        action_18 = np.zeros(18, dtype=np.float32)
        
        # Left arm (0-6) -> (0-6)
        action_18[0:7] = action_16[0:7]
        
        # Left gripper (7) -> (7, 8) - DENORMALIZE from model and duplicate to both fingers
        left_gripper_raw = gripper_normalized_to_raw(action_16[7], self.config)
        action_18[7] = left_gripper_raw
        action_18[8] = left_gripper_raw
        
        # Right arm (8-14) -> (9-15)
        action_18[9:16] = action_16[8:15]
        
        # Right gripper (15) -> (16, 17) - DENORMALIZE from model and duplicate to both fingers
        right_gripper_raw = gripper_normalized_to_raw(action_16[15], self.config)
        action_18[16] = right_gripper_raw
        action_18[17] = right_gripper_raw
        
        msg.position = action_18.tolist()
        self.action_pub.publish(msg)


def run_client(config: ClientConfig, task: str):
    """Main client loop."""
    
    # Initialize ROS2
    rclpy.init()
    ros_node = OpenArmPolicyClientNode(config)
    
    # Spin ROS2 in background
    spin_thread = threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True)
    spin_thread.start()
    
    # Connect to policy server
    policy_client = PolicyClient(config.host, config.port)
    
    # Wait for observations
    logger.info("Waiting for observations from Isaac Sim...")
    while ros_node.get_observation() is None:
        time.sleep(0.1)
    logger.info("✅ Receiving data from Isaac Sim!")
    
    # Inference loop
    logger.info("=" * 60)
    logger.info(f"Starting OpenArm Policy Client")
    logger.info(f"Task: {task}")
    logger.info(f"FPS: {config.fps}")
    logger.info(f"Action chunking: ENABLED (horizon=16)")
    logger.info("=" * 60)
    
    frame_count = 0
    
    try:
        prev_action = None
        while frame_count < config.max_steps:
            loop_start = time.perf_counter()
            
            # Get observation
            obs = ros_node.get_observation()
            if obs is None:
                time.sleep(0.01)
                continue
            
            # Add task prompt
            obs["prompt"] = task
            
            # Run inference via policy server (with action chunking)
            try:
                infer_start = time.perf_counter()
                action_np, is_new_chunk, chunk_step = policy_client.infer(obs, use_chunking=True)
                infer_time = (time.perf_counter() - infer_start) * 1000  # ms
                
                current_state = obs["state"]
                
                # Detailed logging every 30 frames OR when new chunk is fetched
                log_this_frame = (frame_count % 30 == 0) or (is_new_chunk and frame_count > 0)
                
                if log_this_frame:
                    logger.info("=" * 70)
                    if is_new_chunk:
                        logger.info(f"[Frame {frame_count}] 🔄 NEW CHUNK - Inference time: {infer_time:.1f}ms, step={chunk_step}")
                    else:
                        logger.info(f"[Frame {frame_count}] ▶ Using cached action, step={chunk_step} (no inference)")
                    
                    # Current state
                    logger.info(f"Current State (16-dim):")
                    logger.info(f"  Left arm joints:  [{', '.join(f'{x:.3f}' for x in current_state[0:7])}]")
                    logger.info(f"  Left gripper:     {current_state[7]:.4f}")
                    logger.info(f"  Right arm joints: [{', '.join(f'{x:.3f}' for x in current_state[8:15])}]")
                    logger.info(f"  Right gripper:    {current_state[15]:.4f}")
                    
                    # Predicted action
                    logger.info(f"Predicted Action (16-dim):")
                    logger.info(f"  Left arm joints:  [{', '.join(f'{x:.3f}' for x in action_np[0:7])}]")
                    logger.info(f"  Left gripper:     {action_np[7]:.4f}")
                    logger.info(f"  Right arm joints: [{', '.join(f'{x:.3f}' for x in action_np[8:15])}]")
                    logger.info(f"  Right gripper:    {action_np[15]:.4f}")
                    
                    # Delta between state and action
                    delta = action_np - current_state
                    logger.info(f"Delta (Action - State):")
                    logger.info(f"  Left arm delta:   [{', '.join(f'{x:+.3f}' for x in delta[0:7])}]")
                    logger.info(f"  Left gripper Δ:   {delta[7]:+.4f}")
                    logger.info(f"  Right arm delta:  [{', '.join(f'{x:+.3f}' for x in delta[8:15])}]")
                    logger.info(f"  Right gripper Δ:  {delta[15]:+.4f}")
                    
                    # Check for action jitter (if we have previous action)
                    if prev_action is not None:
                        action_change = np.abs(action_np - prev_action)
                        logger.info(f"Action Jitter (vs prev frame): max={action_change.max():.4f}, mean={action_change.mean():.4f}")
                    
                    logger.info(f"Action range: [{action_np.min():.4f}, {action_np.max():.4f}]")
                    logger.info("=" * 70)
                
                prev_action = action_np.copy()
                
                # Publish to Isaac Sim
                ros_node.publish_action(action_np)
                
            except Exception as e:
                logger.warning(f"Inference error: {e}")
                import traceback
                traceback.print_exc()
            
            frame_count += 1
            
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
    parser = argparse.ArgumentParser(description="OpenArm ROS2 Policy Client")
    parser.add_argument(
        "--task", type=str, required=True,
        help="Task description for inference"
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Policy server host"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Policy server port"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Inference frequency (should match training: 30 Hz)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000,
        help="Maximum inference steps"
    )
    
    args = parser.parse_args()
    
    config = ClientConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        max_steps=args.max_steps,
    )
    
    run_client(config, args.task)


if __name__ == "__main__":
    main()
