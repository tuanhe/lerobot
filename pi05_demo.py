import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# Swap this import per-policy
from lerobot.policies.pi05 import PI05Policy

# load a policy
# model_id = "/home/x/Documents/models/lerobot/pi05_base/"  # <- swap checkpoint
model_id = "/home/x/Documents/models/lerobot/pi05_base_migrated/"  # <- swap checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PI05Policy.from_pretrained(model_id).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
# load a lerobotdataset (we will replace with a simpler dataset)
dataset = LeRobotDataset("lerobot/libero")

# pick an episode
episode_index = 0

# each episode corresponds to a contiguous range of frame indices
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx   = dataset.meta.episodes["dataset_to_index"][episode_index]

# get a single frame from that episode (e.g. the first frame)
frame_index = from_idx
frame = dict(dataset[frame_index])

print(f"frame_index : {frame_index}")
print(f"frame length: {len(frame)}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

batch = preprocess(frame)
with torch.inference_mode():
    print(f"frame : {frame}")
    pred_action = policy.select_action(batch)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    print(f"pred_action : {pred_action}")
    pred_action = postprocess(pred_action)
    print(f"pred_action : {pred_action}")
    