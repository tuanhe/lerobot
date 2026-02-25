"""
PI0.5 Inference demo (v2) — uses PI05Processor instead of the pipeline system.

Usage:
    python pi05_demo_v2.py
"""

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from pi05_processor import PI05Processor

# ── config ──────────────────────────────────────────────────────────
model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"
dataset_id = "lerobot/libero"
episode_index = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load policy ─────────────────────────────────────────────────────
policy = PI05Policy.from_pretrained(model_id).to(device).eval()

# ── load processor ──────────────────────────────────────────────────
processor = PI05Processor(
    model_id, device=str(device),
    tokenizer_path="/home/x/Documents/models/paligemma-3b-pt-224/",
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

batch = processor.preprocess(frame)
with torch.inference_mode():
    print(f"frame : {frame}")
    pred_action = policy.select_action(batch)
    print(f"pred_action : {pred_action}")
    pred_action = processor.postprocess(pred_action)
    print(f"pred_action : {pred_action}")
