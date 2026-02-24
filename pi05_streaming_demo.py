"""
PI05 流式推理演示（StreamingPI05）

模式 A（同步，对照）：原始 select_action，队列空时卡 138ms。
模式 B（流式）：StreamingPI05，后台推理，执行几乎不阻塞。
"""

import time
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.policies.pi05.streaming_pi05 import StreamingPI05
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

# ── 配置 ──────────────────────────────────────────────────────────────────────
model_id = "/home/x/Documents/models/lerobot/pi05_base/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 加载模型 ──────────────────────────────────────────────────────────────────
policy = PI05Policy.from_pretrained(model_id).to(device).eval()
cfg = policy.config

# ── 前处理器 ──────────────────────────────────────────────────────────────────
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

# ── 后处理器 ──────────────────────────────────────────────────────────────────
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

# ── 加载数据集（用视频帧模拟连续流）────────────────────────────────────────────
dataset = LeRobotDataset("lerobot/libero")
episode_index = 0
from_idx = int(dataset.meta.episodes["dataset_from_index"][episode_index])
to_idx   = int(dataset.meta.episodes["dataset_to_index"][episode_index])

frames = [dict(dataset[i]) for i in range(from_idx, to_idx)]
print(f"Episode {episode_index}：共 {len(frames)} 帧")
N = cfg.n_action_steps  # 跑一个完整 chunk 的帧数做对比


# ══════════════════════════════════════════════════════════════════════════════
# 模式 A：同步推理（对照组）
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 模式 A：同步推理 ───")
policy.reset()
t0 = time.perf_counter()

for frame in frames[:N]:
    batch = preprocess(frame)
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = postprocess(action)

elapsed_a = (time.perf_counter() - t0) * 1000
print(f"  处理 {N} 帧耗时: {elapsed_a:.1f} ms")
print(f"  最后动作形状: {action.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 模式 B：StreamingPI05 异步推理
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── 模式 B：流式异步推理 ───")
policy.reset()
streamer = StreamingPI05(policy, preprocess, refill_threshold=N // 2)

# 预热：传入第一帧触发首次推理，等队列有内容才进入控制循环
streamer.update_obs(frames[0])
print("  等待首次推理完成（预热）...", end="", flush=True)
while streamer.queue_size() == 0:
    time.sleep(0.005)
print(f" 完成，队列 {streamer.queue_size()} 步")

t0 = time.perf_counter()
blocked_count = 0

for i, frame in enumerate(frames[:N]):
    # 更新观测（后台按需触发新推理）
    streamer.update_obs(frame)

    # 模拟 100 Hz 控制周期（10 ms/帧）
    deadline = t0 + (i + 1) * 0.01

    action = streamer.get_action()
    action = postprocess(action)

    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    else:
        blocked_count += 1  # 超时：推理没跟上执行节奏

elapsed_b = (time.perf_counter() - t0) * 1000
print(f"  处理 {N} 帧耗时: {elapsed_b:.1f} ms")
print(f"  超时帧数（推理未跟上）: {blocked_count}")
print(f"  最后动作形状: {action.shape}")
print(f"  最后队列剩余: {streamer.queue_size()} 步")

print(f"\n  加速比（流式 vs 同步）: {elapsed_a / elapsed_b:.2f}x")
