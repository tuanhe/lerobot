import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

model_id = "/home/x/Documents/models/lerobot/pi05_base/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PI05Policy.from_pretrained(model_id).to(device).eval()
cfg = policy.config

# ── 前处理器 ──────────────────────────────────────────────────────────────
# 1. RenameObservations : 重命名观测键（空映射 = 不改名）
# 2. AddBatchDimension  : 单帧 → batch=1
# 3. Normalizer         : 归一化 STATE/ACTION 到 [-1, 1]（无 stats 则 identity）
# 4. Pi05PrepareState   : 把 state 离散化，拼成 prompt 文本
# 5. Tokenizer          : PaliGemma 分词器，把 prompt → token ids
# 6. DeviceProcessor    : 移到 GPU
preprocess = PolicyProcessorPipeline(
    steps=[
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**cfg.input_features, **cfg.output_features},
            norm_map=cfg.normalization_mapping,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=cfg.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=cfg.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=str(device)),
    ],
)

# ── 后处理器 ──────────────────────────────────────────────────────────────
# 1. Unnormalizer  : 把归一化动作还原回原始尺度
# 2. DeviceProcessor: 移回 CPU
postprocess = PolicyProcessorPipeline(
    steps=[
        UnnormalizerProcessorStep(
            features=cfg.output_features,
            norm_map=cfg.normalization_mapping,
        ),
        DeviceProcessorStep(device="cpu"),
    ],
    to_transition=policy_action_to_transition,
    to_output=transition_to_policy_action,
)

# ── 推理 ──────────────────────────────────────────────────────────────────
dataset = LeRobotDataset("lerobot/libero")
episode_index = 0
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
frame = dict(dataset[from_idx])

print(f"frame_index : {from_idx}")
print(f"frame length: {len(frame)}")

batch = preprocess(frame)
with torch.inference_mode():
    pred_action = policy.select_action(batch)
    print(f"pred_action : {pred_action}")
    pred_action = postprocess(pred_action)
    print(f"pred_action : {pred_action.shape}\n {pred_action}")
