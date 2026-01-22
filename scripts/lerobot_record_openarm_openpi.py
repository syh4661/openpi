#!/usr/bin/env python
"""
LeRobot Dataset Recorder for OpenArm - OpenPI Compatible Version

This script records teleoperation episodes in OpenPI-compatible format:
- 16D state/action format: [left_7_joints, left_gripper, right_7_joints, right_gripper]
- Gripper normalized to [0, 1] where 0=open, 1=closed
- Compatible with Pi0.5 fine-tuning

Run alongside isaac_openarm_teleop_openpi.py.

Usage:
    python lerobot_record_openarm_openpi.py \
        --repo_id=saurabh/openarm_pick_place_v2 \
        --task="Pick up the red cube and place it in the box" \
        --num_episodes=20

VR Controls:
    X button (left): Save episode
    Y button (left): Discard & re-record  
    B button (right): Pause/resume recording
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState, Image, Joy
except ImportError:
    raise ImportError("ROS2 not found. Please source your ROS2 installation.")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

# ============================================================================
# OpenPI 16D Format Definition
# ============================================================================
OPENPI_JOINT_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7", "left_gripper",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7", "right_gripper"
]

OPENPI_STATE_DIM = 16
OPENPI_ACTION_DIM = 16


@dataclass
class RecordOpenPIConfig:
    """Configuration for OpenPI-compatible recording."""
    
    # Dataset settings
    repo_id: str = "saurabh/openarm_dataset_openpi"
    task: str = "Teleoperation task"
    root: str | Path = None  # Uses HF_LEROBOT_HOME if None
    
    # Recording settings
    fps: int = 30
    num_episodes: int = 20
    episode_time_s: float = 60.0
    
    # Video settings
    video: bool = True
    image_writer_threads: int = 4
    
    # OpenPI ROS2 topics (from isaac_openarm_teleop_openpi.py)
    state_topic: str = "/openpi/state"
    action_topic: str = "/openpi/action"
    
    # Camera ROS2 topics
    camera_topics: dict[str, str] = field(default_factory=lambda: {
        "head": "/camera/head/image_raw",
        "wrist_left": "/camera/wrist_left/image_raw",
        "wrist_right": "/camera/wrist_right/image_raw",
    })
    
    # VR button topics
    left_controller_topic: str = "/quest/left_hand/inputs"
    right_controller_topic: str = "/quest/right_hand/inputs"
    
    # Camera resolution
    camera_height: int = 360
    camera_width: int = 480
    
    # Sound feedback
    play_sounds: bool = False


class OpenPIRecorderNode(Node):
    """ROS2 node for recording OpenPI-compatible data."""
    
    def __init__(self, config: RecordOpenPIConfig):
        super().__init__('openpi_recorder')
        self.config = config
        
        # Data buffers
        self.latest_state = None
        self.latest_action = None
        self.latest_images = {cam: None for cam in config.camera_topics}
        self.lock = threading.Lock()
        
        # Button states
        self.button_x = False  # Save
        self.button_y = False  # Discard
        self.button_b = False  # Pause
        
        # QoS profile for sensor data
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        # Subscribe to OpenPI state/action topics
        self.state_sub = self.create_subscription(
            JointState, config.state_topic,
            self._state_callback, qos)
        
        self.action_sub = self.create_subscription(
            JointState, config.action_topic,
            self._action_callback, qos)
        
        # Subscribe to camera topics
        for cam_name, topic in config.camera_topics.items():
            self.create_subscription(
                Image, topic,
                lambda msg, name=cam_name: self._image_callback(msg, name),
                qos)
        
        # Subscribe to controller inputs
        self.create_subscription(
            Joy, config.left_controller_topic,
            self._left_controller_callback, qos)
        self.create_subscription(
            Joy, config.right_controller_topic,
            self._right_controller_callback, qos)
        
        self.get_logger().info("OpenPI Recorder node initialized")
        self.get_logger().info(f"  State topic: {config.state_topic}")
        self.get_logger().info(f"  Action topic: {config.action_topic}")
    
    def _state_callback(self, msg: JointState):
        if len(msg.position) >= OPENPI_STATE_DIM:
            with self.lock:
                self.latest_state = np.array(msg.position[:OPENPI_STATE_DIM], dtype=np.float32)
    
    def _action_callback(self, msg: JointState):
        if len(msg.position) >= OPENPI_ACTION_DIM:
            with self.lock:
                self.latest_action = np.array(msg.position[:OPENPI_ACTION_DIM], dtype=np.float32)
    
    def _image_callback(self, msg: Image, cam_name: str):
        try:
            # Convert ROS Image to numpy array
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
            elif msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
                img = img[:, :, ::-1]  # BGR to RGB
            else:
                return  # Unsupported encoding
            
            with self.lock:
                self.latest_images[cam_name] = img.copy()
        except Exception as e:
            self.get_logger().warn(f"Image error ({cam_name}): {e}")
    
    def _left_controller_callback(self, msg: Joy):
        # X = buttons[0], Y = buttons[1]
        if len(msg.buttons) >= 2:
            self.button_x = msg.buttons[0] == 1  # Save
            self.button_y = msg.buttons[1] == 1  # Discard
    
    def _right_controller_callback(self, msg: Joy):
        # B = buttons[1]
        if len(msg.buttons) >= 2:
            self.button_b = msg.buttons[1] == 1  # Pause
    
    def get_observation(self) -> dict:
        """Get current observation in OpenPI format."""
        with self.lock:
            obs = {
                "state": self.latest_state.copy() if self.latest_state is not None else None,
                "images": {k: v.copy() if v is not None else None 
                          for k, v in self.latest_images.items()}
            }
        return obs
    
    def get_action(self) -> Optional[np.ndarray]:
        """Get current action in OpenPI format."""
        with self.lock:
            return self.latest_action.copy() if self.latest_action is not None else None
    
    def get_buttons(self) -> dict:
        """Get button states."""
        return {
            "save": self.button_x,
            "discard": self.button_y,
            "pause": self.button_b,
        }
    
    def has_data(self) -> bool:
        """Check if we're receiving data."""
        return self.latest_state is not None


def create_openpi_dataset_features(config: RecordOpenPIConfig) -> dict:
    """Create OpenPI-compatible dataset features."""
    features = {}
    
    # Observation state (16D) - OpenPI format
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (OPENPI_STATE_DIM,),
        "names": OPENPI_JOINT_NAMES,
    }
    
    # Action (16D) - OpenPI format
    features["action"] = {
        "dtype": "float32",
        "shape": (OPENPI_ACTION_DIM,),
        "names": OPENPI_JOINT_NAMES,
    }
    
    # Camera images
    for cam_name in config.camera_topics:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video" if config.video else "image",
            "shape": (config.camera_height, config.camera_width, 3),
            "names": ["height", "width", "channels"],
        }
    
    return features


def record_episode(
    node: OpenPIRecorderNode,
    dataset: LeRobotDataset,
    config: RecordOpenPIConfig,
) -> tuple[bool, str]:
    """Record a single episode.
    
    Returns:
        (success, reason): Whether episode was saved and why it ended.
    """
    start_time = time.perf_counter()
    frame_count = 0
    paused = False
    prev_pause = False
    
    print(f"[Recording] Episode started - Max {config.episode_time_s}s")
    print("[Controls] X=Save, Y=Discard, B=Pause/Resume")
    
    while True:
        loop_start = time.perf_counter()
        
        # Spin ROS to get latest data
        rclpy.spin_once(node, timeout_sec=0.001)
        
        # Check buttons
        buttons = node.get_buttons()
        
        # X button: Save episode
        if buttons["save"]:
            print("[Recording] Saving episode...")
            return True, "saved"
        
        # Y button: Discard episode
        if buttons["discard"]:
            print("[Recording] Discarding episode...")
            return False, "discarded"
        
        # B button: Toggle pause
        if buttons["pause"] and not prev_pause:
            paused = not paused
            print(f"[Recording] {'PAUSED' if paused else 'RESUMED'}")
        prev_pause = buttons["pause"]
        
        if paused:
            time.sleep(0.05)
            continue
        
        # Get observation and action
        obs = node.get_observation()
        action = node.get_action()
        
        if obs["state"] is None or action is None:
            time.sleep(0.01)
            continue
        
        # Build frame
        frame = {
            "observation.state": obs["state"],
            "action": action,
        }
        
        # Add camera images
        for cam_name in config.camera_topics:
            img = obs["images"].get(cam_name)
            if img is not None:
                frame[f"observation.images.{cam_name}"] = img
            else:
                # Black placeholder
                frame[f"observation.images.{cam_name}"] = np.zeros(
                    (config.camera_height, config.camera_width, 3), dtype=np.uint8
                )
        
        # Add frame to dataset
        dataset.add_frame(frame, task=config.task)
        frame_count += 1
        
        # Status print every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"[Recording] {frame_count} frames, {elapsed:.1f}s")
        
        # Check max duration
        elapsed = time.perf_counter() - start_time
        if elapsed >= config.episode_time_s:
            print(f"[Recording] Max duration reached ({frame_count} frames)")
            return True, "max_duration"
        
        # Maintain FPS
        dt = time.perf_counter() - loop_start
        sleep_time = 1.0 / config.fps - dt
        if sleep_time > 0:
            time.sleep(sleep_time)


def record_dataset(config: RecordOpenPIConfig) -> LeRobotDataset:
    """Main recording function."""
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("OpenPI Dataset Recorder")
    print("=" * 60)
    print(f"Repo: {config.repo_id}")
    print(f"Task: {config.task}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Format: 16D OpenPI (7+1+7+1)")
    print("=" * 60)
    
    # Initialize ROS2
    rclpy.init()
    node = OpenPIRecorderNode(config)
    
    # Create dataset
    features = create_openpi_dataset_features(config)
    
    root = Path(config.root) if config.root else None
    
    dataset = LeRobotDataset.create(
        repo_id=config.repo_id,
        fps=config.fps,
        root=root,
        robot_type="openarm_bimanual",
        features=features,
        use_videos=config.video,
        image_writer_threads=config.image_writer_threads,
    )
    
    print(f"[Dataset] Created at: {dataset.root}")
    
    try:
        # Wait for teleop data
        print("\n[Waiting] Waiting for teleop data...")
        print("Please ensure isaac_openarm_teleop_openpi.py is running.")
        
        wait_start = time.time()
        last_print = 0
        while not node.has_data():
            rclpy.spin_once(node, timeout_sec=0.1)
            elapsed = time.time() - wait_start
            if elapsed - last_print >= 5.0:
                print(f"[Waiting] Still waiting... ({int(elapsed)}s)")
                last_print = elapsed
            time.sleep(0.1)
        
        print("[Ready] Receiving teleop data!")
        print("\n" + "=" * 60)
        print("RECORDING READY")
        print("Press B button to start each episode")
        print("=" * 60)
        
        # Record episodes
        recorded = 0
        while recorded < config.num_episodes:
            print(f"\n[Episode {recorded + 1}/{config.num_episodes}] Press B to start...")
            
            # Wait for B button press
            while True:
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.button_b:
                    # Wait for release
                    while node.button_b:
                        rclpy.spin_once(node, timeout_sec=0.01)
                    break
            
            # Record episode
            success, reason = record_episode(node, dataset, config)
            
            if success:
                dataset.save_episode()
                recorded += 1
                print(f"[Episode {recorded}] Saved ({reason})")
            else:
                dataset.clear_episode_buffer()
                print(f"[Episode] Discarded ({reason})")
        
        print("\n" + "=" * 60)
        print(f"RECORDING COMPLETE! {recorded} episodes saved.")
        print("=" * 60)
        
    finally:
        node.destroy_node()
        rclpy.shutdown()
        # dataset.finalize() # Not available in lerobot 0.3.3
    
    return dataset


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Record OpenPI-compatible teleoperation episodes"
    )
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Dataset repository ID (e.g., saurabh/openarm_pick_v2)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task description for language conditioning")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of episodes to record (default: 20)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording FPS (default: 30)")
    parser.add_argument("--episode_time_s", type=float, default=60.0,
                        help="Max episode duration in seconds (default: 60)")
    parser.add_argument("--root", type=str, default=None,
                        help="Dataset root directory")
    parser.add_argument("--no_video", action="store_true",
                        help="Save images instead of videos")
    
    args = parser.parse_args()
    
    config = RecordOpenPIConfig(
        repo_id=args.repo_id,
        task=args.task,
        num_episodes=args.num_episodes,
        fps=args.fps,
        episode_time_s=args.episode_time_s,
        root=args.root,
        video=not args.no_video,
    )
    
    record_dataset(config)


if __name__ == "__main__":
    main()
