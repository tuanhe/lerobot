"""
PI0.5 deployment demo — input from camera frames + user task, not dataset.

Usage:
    python pi05_deploy.py

This script shows how to run PI0.5 inference with raw numpy/BGR images
(as you'd get from cv2.VideoCapture) and a user-provided task string,
instead of loading from a LeRobotDataset.
"""

import cv2
import numpy as np
import torch
from lerobot.policies.pi05 import PI05Policy
from pi05_processor import PI05Processor

# ── config ──────────────────────────────────────────────────────────
model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"
tokenizer_path = "/home/x/Documents/models/paligemma-3b-pt-224/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load policy & processor ─────────────────────────────────────────
policy = PI05Policy.from_pretrained(model_id).to(device).eval()
processor = PI05Processor(model_id, device=str(device), tokenizer_path=tokenizer_path)


def images_to_tensor(bgr_image: np.ndarray, size: tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Convert a BGR numpy image (from cv2) to a [C, H, W] float32 tensor in [0, 1]."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def predict(
    image1: np.ndarray,
    image2: np.ndarray,
    state: np.ndarray,
    task: str,
) -> np.ndarray:
    """Run one inference step.

    Args:
        image1: BGR image from camera 1, any resolution.
        image2: BGR image from camera 2, any resolution.
        state:  Robot joint state, float array of shape (8,).
        task:   Natural language task description.

    Returns:
        Action as numpy array of shape (N,).
    """
    frame = {
        "observation.images.image": images_to_tensor(image1),
        "observation.images.image2": images_to_tensor(image2),
        "observation.state": torch.from_numpy(state).float(),
        "task": task,
    }

    batch = processor.preprocess(frame)
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = processor.postprocess(action)
    return action.squeeze(0).numpy()


# ── demo: use dataset frames to simulate camera input ───────────────
if __name__ == "__main__":
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Load one frame from dataset as ground truth for comparison
    dataset = LeRobotDataset("lerobot/libero")
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    frame = dict(dataset[from_idx])

    # Simulate what you'd get from cameras:
    # convert [C,H,W] float RGB tensor → [H,W,C] uint8 BGR numpy (like cv2 output)
    def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
        rgb = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    img1_bgr = tensor_to_bgr(frame["observation.images.image"])
    img2_bgr = tensor_to_bgr(frame["observation.images.image2"])
    state = frame["observation.state"].numpy()
    task = frame["task"]

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    action = predict(img1_bgr, img2_bgr, state, task)
    print(f"task   : {task}")
    print(f"state  : {state}")
    print(f"action : {action}")
