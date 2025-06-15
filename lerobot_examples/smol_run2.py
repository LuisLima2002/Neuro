from pathlib import Path
import numpy as np
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import torch
from copy import copy
from contextlib import nullcontext
from lerobot.common.utils.utils import (
    get_safe_torch_device,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from typing import Any
from lerobot.common.cameras.configs import ColorMode, Cv2Rotation
import time
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}
def build_dataset_frame(
    values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    ds_features ={'action': {'dtype': 'float32', 'shape': (6,), 'names': ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']}, 'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']}, 'observation.images.robo': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}, 'observation.images.out': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}, 'timestamp': {'dtype': 'float32', 'shape': (1,), 'names': None}, 'frame_index': {'dtype': 'int64', 'shape': (1,), 'names': None}, 'episode_index': {'dtype': 'int64', 'shape': (1,), 'names': None}, 'index': {'dtype': 'int64', 'shape': (1,), 'names': None}, 'task_index': {'dtype': 'int64', 'shape': (1,), 'names': None}}
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]

    return frame

def predict_action(
    observation: dict[str, np.ndarray],
    policy: SmolVLAPolicy,
    device: torch.device,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            observation[name] = torch.from_numpy(observation[name])
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action

if __name__=="__main__":
    from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower
    camera_external_config = OpenCVCameraConfig(
        index_or_path=7,
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    
    camera_bot_config = OpenCVCameraConfig(
        index_or_path=1,
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
    )

    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM1",
        cameras={"robo": camera_bot_config, "out": camera_external_config},
        id="movie",
    )
    robot = SO101Follower(robot_config)
    robot.connect()

    device = get_safe_torch_device("cuda")
    policy = SmolVLAPolicy.from_pretrained("/home/fablab/lerobot/outputs/train/Neurof14K/checkpoints/last/pretrained_model")
    control_time_s=40
    fps=30
    start_episode_t = time.perf_counter()
    timestamp = 0 
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        observation = robot.get_observation()
        # Predict the next action with respect to the current observation
        observation_frame = build_dataset_frame(observation, prefix="observation")
        action_values = predict_action(observation_frame,policy,device,policy.config.use_amp,task="pick the yellow screwdriver and hold it in the air",robot_type=robot.robot_type)
        action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}

        # Send the action to the robot
        robot.send_action(action)
        # Check if the episode is done
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t
        print(timestamp)

    # robot.disconnect() 
