"""
StreamingPI05 —— 面向连续视频流的异步推理封装

核心思想：
  原始 select_action 在队列空时同步等待推理（约 138ms 卡顿）。
  StreamingPI05 在队列剩余 refill_threshold 步时就在后台线程触发新推理，
  让 GPU 推理和机器人执行真正并行，消除执行停顿。

时间轴对比：
  原始：[exec 0-49] → 卡 138ms → [exec 50-99] → 卡 138ms → ...
  流式：[exec 0-49]                [exec 25-74]               ...
           ↑ 队列剩 25 步时触发后台推理 ↑ 新 chunk 已就绪，无停顿

典型控制循环：
    streamer = StreamingPI05(policy, preprocess)
    streamer.update_obs(get_first_frame())   # 触发第一次推理（预热）

    while running:
        frame = robot.get_observation()
        streamer.update_obs(frame)           # 更新观测，必要时触发推理
        action = streamer.get_action()       # 几乎立即返回
        robot.execute(action)
"""

import logging
import threading
import time
from collections import deque

import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class StreamingPI05:
    """
    PI05Policy 的异步推理包装器，适用于连续视频流场景。

    Attributes:
        policy:            底层 PI05Policy，推理在后台线程中调用。
        preprocess:        前处理器 PolicyProcessorPipeline。
        refill_threshold:  队列剩余步数低于此值时触发新推理。
                           建议设为 n_action_steps 的 1/4 到 1/2。
    """

    def __init__(
        self,
        policy: PI05Policy,
        preprocess,
        refill_threshold: int | None = None,
    ):
        self.policy = policy
        self.preprocess = preprocess
        self._n_action_steps = policy.config.n_action_steps

        # 默认阈值：n_action_steps 的一半，给推理留足裕量
        self._refill_threshold = refill_threshold or (self._n_action_steps // 2)

        self._action_queue: deque = deque()
        self._lock = threading.Lock()
        self._inference_running = False
        self._latest_batch = None

    # ──────────────────────────────── 公开接口 ────────────────────────────────

    def update_obs(self, frame: dict) -> None:
        """
        每个控制周期调用，传入最新的原始观测帧。

        会在 CPU 上同步完成前处理，然后在必要时异步触发 GPU 推理。
        """
        batch = self.preprocess(frame)
        with self._lock:
            self._latest_batch = batch
        self._maybe_trigger_inference()

    def get_action(self) -> torch.Tensor:
        """
        返回下一步动作张量，形状 [1, action_dim]。

        正常情况几乎立即返回；仅在极端情况（队列意外耗尽）短暂自旋等待。
        """
        while True:
            with self._lock:
                if self._action_queue:
                    return self._action_queue.popleft()
            # 队列暂时为空：推理应正在进行，自旋等待（罕见）
            time.sleep(0.001)

    def queue_size(self) -> int:
        """返回当前队列中剩余的动作步数。"""
        with self._lock:
            return len(self._action_queue)

    def is_inference_running(self) -> bool:
        """返回后台推理线程是否正在运行。"""
        with self._lock:
            return self._inference_running

    def reset(self):
        """新 episode 开始时调用，清空队列和缓存观测。"""
        with self._lock:
            self._action_queue.clear()
            self._latest_batch = None
            # 不重置 _inference_running：正在运行的推理线程会自行结束

    # ──────────────────────────────── 内部逻辑 ────────────────────────────────

    def _maybe_trigger_inference(self) -> None:
        """满足条件时启动后台推理线程（持锁检查，避免重复启动）。"""
        with self._lock:
            if self._inference_running:
                return  # 已有推理在跑，不重复触发
            if len(self._action_queue) >= self._refill_threshold:
                return  # 队列还充裕，暂不触发
            if self._latest_batch is None:
                return  # 还没收到第一帧，等待
            self._inference_running = True
            batch = self._latest_batch  # 快照当前最新帧

        thread = threading.Thread(
            target=self._run_inference,
            args=(batch,),
            daemon=True,  # 主进程退出时自动终止
        )
        thread.start()

    def _run_inference(self, batch) -> None:
        """后台线程：运行推理并将结果追加到队列。"""
        try:
            # predict_action_chunk 返回 [1, chunk_size, action_dim]
            actions = self.policy.predict_action_chunk(batch)
            # 截取 n_action_steps 步，转为列表方便 extend
            # transpose: [1, n, d] → [n, 1, d]，每项形状 [1, action_dim]
            actions = actions[:, : self._n_action_steps]
            new_actions = list(actions.transpose(0, 1))
        except Exception:
            logging.exception("StreamingPI05: 后台推理出错")
            new_actions = []

        with self._lock:
            self._action_queue.extend(new_actions)
            self._inference_running = False

        # 推理结束后再检查一次：应对高频场景下队列消耗快于推理的情况
        self._maybe_trigger_inference()
