"""pi05 推理速度对比测试"""
import time
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
)

model_id = "/home/x/Documents/models/lerobot/pi05_base/"
device = torch.device("cuda")


def load_policy():
    policy = PI05Policy.from_pretrained(model_id).to(device).eval()
    cfg = policy.config

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

    return policy, preprocess


def get_batch(preprocess):
    dataset = LeRobotDataset("lerobot/libero")
    frame = dict(dataset[dataset.meta.episodes["dataset_from_index"][0]])
    return preprocess(frame)


def benchmark(policy, batch, n=10, label=""):
    for _ in range(3):
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n):
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / n * 1000
    print(f"  {label:<35} {avg_ms:>7.1f} ms   {1000/avg_ms:>5.1f} Hz")
    return avg_ms


print("\n加载模型中...")
policy, preprocess = load_policy()
batch = get_batch(preprocess)

print(f"\n{'='*65}")
print(f"  {'配置':<35} {'耗时':>10}   {'频率':>8}")
print(f"{'='*65}")

base_ms = benchmark(policy, batch, label="float32 (优化后基线)")

policy.enable_bfloat16_inference()
bf16_ms = benchmark(policy, batch, label="bfloat16")

print(f"{'='*65}")
print(f"  原始基线 (优化前):               ~217.6 ms    4.6 Hz")
print(f"  bfloat16 相对优化后基线加速: {base_ms/bf16_ms:.2f}x")
print(f"  bfloat16 相对原始基线加速:   {217.6/bf16_ms:.2f}x")
print(f"{'='*65}\n")
