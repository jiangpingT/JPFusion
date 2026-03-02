"""
world_model_env.py — 世界模型 Gymnasium 环境（Phase 4）

WorldModelEnv:       MLP Ensemble 包装（原版，保留向后兼容）
DeepONetWorldModelEnv: FusionDeepONet 包装（Path C，历史感知，无递归误差）

RL Agent 在世界模型里训练，不需要真实仿真器（Dyna 循环的核心）。
不确定性高的状态 → 惩罚探索（避免 model exploitation）
不确定性低的状态 → 正常奖励（安全区域）
"""

from collections import deque
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from backend.rl.world_model import WorldModelEnsemble, get_world_model
from backend.rl.disruption import check_disruption
from backend.rl.rewards import LAWSON_TARGET

# 不确定性惩罚权重：不确定性过高 → 惩罚（防止模型利用 world model 漏洞）
UNCERTAINTY_PENALTY = -5.0
UNCERTAINTY_THRESHOLD = 0.1  # 超过此值的不确定性才施加惩罚


class WorldModelEnv(gym.Env):
    """
    World Model 包装的 Gymnasium 环境（Dyna 用）

    与 FusionEnv 接口完全一致，只是动力学由 WorldModelEnsemble 提供。
    额外信号：uncertainty（不确定性），用于 agent 判断是否在已知区域内探索。
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, world_model: WorldModelEnsemble = None,
                 max_steps: int = 500, render_mode: Optional[str] = None):
        super().__init__()
        self.world_model = world_model or get_world_model()
        if self.world_model is None:
            raise RuntimeError("WorldModelEnsemble 未初始化，请先训练世界模型")

        self.max_steps   = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(3,), dtype=np.float32
        )

        self._state      = None
        self._step_count = 0
        self._ep_reward  = 0.0
        self._ep_uncertainty = []

    def _sample_initial_state(self) -> np.ndarray:
        """随机采样初始状态（从 FusionEnv 借用，保持一致）"""
        from backend.rl.fusion_env import FusionEnv
        tmp = FusionEnv()
        tmp.reset(seed=int(self.np_random.integers(0, 100000)))
        return tmp._state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state      = self._sample_initial_state()
        self._step_count = 0
        self._ep_reward  = 0.0
        self._ep_uncertainty = []

        return self._state.copy(), {"uncertainty": 0.0}

    def step(self, action: np.ndarray):
        action = np.clip(action, -0.1, 0.1).astype(np.float32)

        # ─── 世界模型预测 ───────────────────────────────────────────────
        next_state_mean, next_state_std, reward_mean, uncertainty = \
            self.world_model.predict_next(self._state, action)

        self._ep_uncertainty.append(uncertainty)

        # ─── 从 ensemble 分布中采样下一步（训练时增加多样性）────────────
        # 采样：next_state = mean + std * noise（添加随机性，防止确定性循环）
        noise = self.np_random.standard_normal(next_state_mean.shape).astype(np.float32)
        next_state = np.clip(next_state_mean + 0.1 * next_state_std * noise, 0.0, 1.0)

        # ─── 不确定性惩罚（防止 agent 利用 world model 漏洞）───────────
        unc_penalty = 0.0
        if uncertainty > UNCERTAINTY_THRESHOLD:
            unc_penalty = UNCERTAINTY_PENALTY * (uncertainty - UNCERTAINTY_THRESHOLD)

        total_reward = reward_mean + unc_penalty

        # ─── 破裂检测 ────────────────────────────────────────────────────
        disrupted, disruption_reason = check_disruption(next_state)
        if disrupted:
            from backend.rl.rewards import compute_disruption_penalty
            total_reward = compute_disruption_penalty()

        self._state       = next_state.astype(np.float32)
        self._step_count += 1
        self._ep_reward  += total_reward

        terminated = disrupted
        truncated  = self._step_count >= self.max_steps

        info = {
            "uncertainty":      uncertainty,
            "unc_penalty":      unc_penalty,
            "reward_mean":      reward_mean,
            "disrupted":        disrupted,
            "next_state_std":   next_state_std.mean(),
        }

        return self._state.copy(), float(total_reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return (f"[WM] Step {self._step_count:4d} | "
                    f"unc={self._ep_uncertainty[-1]:.4f} if self._ep_uncertainty else 0.0")

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# DeepONet 世界模型环境（Path C）
# ─────────────────────────────────────────────────────────────────────────────

class DeepONetWorldModelEnv(gym.Env):
    """
    FusionDeepONet 世界模型包装的 Gymnasium 环境（Dyna 用）

    与 WorldModelEnv 接口完全一致（observation_space / action_space / reset / step）。

    核心区别：
      - _initial_state（s_0）：episode 起点，整个 episode 不变，作为 Branch 锚点
      - _action_history：deque(maxlen=K)，记录过去 K 步动作
      - step() 始终用 s_0 作 branch 输入 → 无递归误差累积
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, deeponet=None, max_steps: int = 500,
                 render_mode: Optional[str] = None):
        super().__init__()

        if deeponet is None:
            from backend.rl.world_model_deeponet import load_deeponet_ensemble
            deeponet = load_deeponet_ensemble()
        if deeponet is None:
            raise RuntimeError(
                "FusionDeepONetEnsemble 未初始化，请先运行 train_deeponet_world_model()"
            )

        self.deeponet    = deeponet
        self.max_steps   = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(3,), dtype=np.float32
        )

        from backend.rl.world_model_deeponet import K as _K
        self._K              = _K
        self._state          = None
        self._initial_state  = None
        self._action_history = None
        self._step_count     = 0
        self._ep_reward      = 0.0
        self._ep_uncertainty = []

    def _sample_initial_state(self) -> np.ndarray:
        from backend.rl.fusion_env import FusionEnv
        tmp = FusionEnv()
        tmp.reset(seed=int(self.np_random.integers(0, 100000)))
        return tmp._state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s0 = self._sample_initial_state()
        self._initial_state  = s0.copy()
        self._state          = s0.copy()
        self._action_history = deque(
            [np.zeros(3, dtype=np.float32)] * self._K, maxlen=self._K
        )
        self._step_count     = 0
        self._ep_reward      = 0.0
        self._ep_uncertainty = []
        return self._state.copy(), {"uncertainty": 0.0}

    def step(self, action: np.ndarray):
        action = np.clip(action, -0.1, 0.1).astype(np.float32)
        self._action_history.append(action.copy())

        # DeepONet 预测（Branch 始终以 s_0 为锚点，不用前一步预测值）
        next_state_mean, next_state_std, reward_mean, uncertainty = \
            self.deeponet.predict_next(
                self._initial_state,
                self._action_history,
                self._step_count,
            )

        self._ep_uncertainty.append(uncertainty)

        # 从 ensemble 分布采样（训练时增加多样性）
        noise = self.np_random.standard_normal(next_state_mean.shape).astype(np.float32)
        next_state = np.clip(next_state_mean + 0.1 * next_state_std * noise, 0.0, 1.0)

        # 不确定性惩罚（防止 agent 利用 world model 漏洞）
        unc_penalty = 0.0
        if uncertainty > UNCERTAINTY_THRESHOLD:
            unc_penalty = UNCERTAINTY_PENALTY * (uncertainty - UNCERTAINTY_THRESHOLD)

        total_reward = reward_mean + unc_penalty

        # 破裂检测
        disrupted, disruption_reason = check_disruption(next_state)
        if disrupted:
            from backend.rl.rewards import compute_disruption_penalty
            total_reward = compute_disruption_penalty()

        self._state       = next_state.astype(np.float32)
        self._step_count += 1
        self._ep_reward  += total_reward

        terminated = disrupted
        truncated  = self._step_count >= self.max_steps

        info = {
            "uncertainty":    uncertainty,
            "unc_penalty":    unc_penalty,
            "reward_mean":    reward_mean,
            "disrupted":      disrupted,
            "next_state_std": float(next_state_std.mean()),
        }

        return self._state.copy(), float(total_reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            unc = self._ep_uncertainty[-1] if self._ep_uncertainty else 0.0
            return (f"[DeepONet WM] Step {self._step_count:4d} | "
                    f"unc={unc:.4f}")

    def close(self):
        pass
