"""
PI0.5 Processor — self-contained pre/post processing for inference.

Replaces the generic pipeline system (Registry → JSON → dynamic class
resolution → EnvTransition wrapping/unwrapping across 10+ files) with
one class and two methods.

Usage:
    from pi05_processor import PI05Processor

    processor = PI05Processor(model_path, device="cuda")
    batch      = processor.preprocess(frame)
    action_out = processor.postprocess(action)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer


class PI05Processor:
    """Self-contained pre/post processor for PI0.5 inference.

    preprocess(frame)   : raw dataset frame  →  model-ready batch
    postprocess(action) : model output tensor →  unnormalized action on CPU

    All logic is inline. No registry, no intermediate representation,
    no JSON-driven step composition. Read top-to-bottom to see what
    happens to your data.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        tokenizer_path: str | None = None,
    ):
        self.device = device
        self.model_path = Path(model_path)

        # ── load configs from pretrained JSON ──
        pre_cfg = self._load_json("policy_preprocessor.json")
        post_cfg = self._load_json("policy_postprocessor.json")

        # Normalization settings  (feature_type → mode, e.g. {"STATE": "MIN_MAX"})
        pre_norm_step = self._find_step_config(pre_cfg, "normalizer_processor")
        post_norm_step = self._find_step_config(post_cfg, "unnormalizer_processor")

        self.pre_norm_map = pre_norm_step.get("norm_map", {})
        self.pre_features = pre_norm_step.get("features", {})
        self.post_norm_map = post_norm_step.get("norm_map", {})
        self.post_features = post_norm_step.get("features", {})

        # Normalization stats from safetensors (empty dict if file doesn't exist)
        self.pre_stats = self._load_stats("normalizer_processor.safetensors")
        self.post_stats = self._load_stats("unnormalizer_processor.safetensors")

        # State tokenizer settings
        state_tok = self._find_step_config(pre_cfg, "pi05_prepare_state_tokenizer_processor_step")
        self.max_state_dim = state_tok.get("max_state_dim", 32)

        # Language tokenizer
        tok_cfg = self._find_step_config(pre_cfg, "tokenizer_processor")
        self.tokenizer_max_length = tok_cfg.get("max_length", 200)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or tok_cfg.get("tokenizer_name", "google/paligemma-3b-pt-224")
        )

    # ────────────────────────────────────────────────────────────────
    #  preprocess:  frame dict  →  model-ready batch dict
    # ────────────────────────────────────────────────────────────────

    def preprocess(self, frame: dict) -> dict:
        # 1) Add batch dimension to every tensor
        batch = {
            k: v.unsqueeze(0) if isinstance(v, Tensor) else v
            for k, v in frame.items()
        }

        # 2) Normalize observations (skip images / actions, only state)
        for key, feat in self.pre_features.items():
            if feat["type"] != "ACTION" and key in batch:
                batch[key] = self._normalize(
                    batch[key], key, feat["type"],
                    self.pre_norm_map, self.pre_stats,
                )

        # 3) State → pad → discretize → text prompt → tokens
        state = batch.get("observation.state")
        if state is not None:
            state = _pad(state, self.max_state_dim)
            batch["observation.state"] = state

        task = batch.get("task", "")
        tasks = [task] if isinstance(task, str) else list(task)

        prompts = _build_prompts(state, tasks)
        tokens = self.tokenizer(
            prompts,
            max_length=self.tokenizer_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        batch["observation.language.tokens"] = tokens["input_ids"]
        batch["observation.language.attention_mask"] = tokens["attention_mask"].to(torch.bool)

        # 4) Move tensors to device
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    # ────────────────────────────────────────────────────────────────
    #  postprocess:  model action tensor  →  unnormalized action (CPU)
    # ────────────────────────────────────────────────────────────────

    def postprocess(self, action: Tensor) -> Tensor:
        action = self._unnormalize(
            action, "action", "ACTION",
            self.post_norm_map, self.post_stats,
        )
        return action.cpu()

    # ────────────────────────────────────────────────────────────────
    #  normalization
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(x, key, feat_type, norm_map, stats):
        mode = norm_map.get(feat_type, "IDENTITY")
        if mode == "IDENTITY" or key not in stats:
            return x
        return _apply_norm(x, stats[key], mode, inverse=False)

    @staticmethod
    def _unnormalize(x, key, feat_type, norm_map, stats):
        mode = norm_map.get(feat_type, "IDENTITY")
        if mode == "IDENTITY" or key not in stats:
            return x
        return _apply_norm(x, stats[key], mode, inverse=True)

    # ────────────────────────────────────────────────────────────────
    #  file loading helpers
    # ────────────────────────────────────────────────────────────────

    def _load_json(self, filename: str) -> dict:
        with open(self.model_path / filename) as f:
            return json.load(f)

    def _load_stats(self, filename: str) -> dict[str, dict[str, Tensor]]:
        """Load normalizer stats from safetensors. Returns {} if file missing."""
        path = self.model_path / filename
        if not path.exists():
            return {}
        from safetensors.torch import load_file
        raw = load_file(str(path))
        stats: dict[str, dict[str, Tensor]] = {}
        for flat_key, tensor in raw.items():
            feature_key, stat_name = flat_key.rsplit(".", 1)
            stats.setdefault(feature_key, {})[stat_name] = tensor.to(torch.float32)
        return stats

    @staticmethod
    def _find_step_config(pipeline_cfg: dict, registry_name: str) -> dict:
        for step in pipeline_cfg.get("steps", []):
            if step.get("registry_name") == registry_name:
                return step.get("config", {})
        return {}


# ════════════════════════════════════════════════════════════════════
#  Pure helper functions
# ════════════════════════════════════════════════════════════════════

def _pad(x: Tensor, target_dim: int) -> Tensor:
    """Pad (or truncate) last dimension to target_dim."""
    d = x.shape[-1]
    if d >= target_dim:
        return x[..., :target_dim]
    zeros = torch.zeros(*x.shape[:-1], target_dim - d, device=x.device, dtype=x.dtype)
    return torch.cat([x, zeros], dim=-1)


def _build_prompts(state: Tensor, tasks: list[str]) -> list[str]:
    """Discretize normalized state into 256 bins, build prompt strings."""
    state_np = state.cpu().numpy()
    bins = np.linspace(-1, 1, 257)[:-1]          # 256 bin edges
    prompts = []
    for i, task in enumerate(tasks):
        disc = np.digitize(state_np[i], bins) - 1
        state_str = " ".join(map(str, disc))
        text = task.strip().replace("_", " ").replace("\n", " ")
        prompts.append(f"Task: {text}, State: {state_str};\nAction: ")
    return prompts


_EPS = 1e-8

def _apply_norm(x: Tensor, stats: dict[str, Tensor], mode: str, *, inverse: bool) -> Tensor:
    """Apply or invert normalization. Supports MEAN_STD / MIN_MAX / QUANTILES."""
    if mode == "MEAN_STD":
        mean, std = stats["mean"].to(x.device), stats["std"].to(x.device)
        if inverse:
            return x * std + mean
        return (x - mean) / (std + _EPS)

    # MIN_MAX and QUANTILES share the same formula, just different stat keys
    if mode == "MIN_MAX":
        lo, hi = stats["min"].to(x.device), stats["max"].to(x.device)
    elif mode == "QUANTILES":
        lo, hi = stats["q01"].to(x.device), stats["q99"].to(x.device)
    else:
        return x  # unknown mode → identity

    denom = hi - lo
    denom = torch.where(denom == 0, torch.tensor(_EPS, device=x.device), denom)
    if inverse:
        return (x + 1.0) * denom / 2.0 + lo
    return 2.0 * (x - lo) / denom - 1.0
