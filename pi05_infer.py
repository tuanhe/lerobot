"""
PI0.5 Inference — minimal, self-contained example.

Usage:
    python pi05_infer.py
"""

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

# ── config ──────────────────────────────────────────────────────────
model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"
dataset_id = "lerobot/libero"
episode_index = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load policy ─────────────────────────────────────────────────────
policy = PI05Policy.from_pretrained(model_id).to(device).eval()

# ── load pre/post processors directly from pretrained JSON configs ──
preprocess = PolicyProcessorPipeline.from_pretrained(
    model_id,
    config_filename="policy_preprocessor.json",
    overrides={"device_processor": {"device": str(device)}},
    to_transition=batch_to_transition,
    to_output=transition_to_batch,
)
postprocess = PolicyProcessorPipeline.from_pretrained(
    model_id,
    config_filename="policy_postprocessor.json",
    to_transition=policy_action_to_transition,
    to_output=transition_to_policy_action,
)

# ── load dataset & pick a frame ─────────────────────────────────────
dataset = LeRobotDataset(dataset_id)
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
frame = dict(dataset[from_idx])

print(f"frame_index : {from_idx}")
print(f"frame length: {len(frame)}")

# ── inference ───────────────────────────────────────────────────────
torch.manual_seed(42)
torch.cuda.manual_seed(42)

batch = preprocess(frame)
with torch.inference_mode():
    print(f"frame : {frame}")
    pred_action = policy.select_action(batch)
    print(f"pred_action : {pred_action}")
    pred_action = postprocess(pred_action)
    print(f"pred_action : {pred_action}")
