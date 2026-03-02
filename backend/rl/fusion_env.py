"""
fusion_env.py — FusionEnv: 托卡马克等离子体控制 Gymnasium 环境

完全类比 JPRobot 的 PyBullet 仿真：
  - 状态空间（7维）：等离子体物理参数（归一化 0~1）
  - 动作空间（3维连续）：加热功率 / 燃料注入 / 电流 增量
  - 奖励：Lawson 准则驱动
  - 终止：4个破裂条件（类比机器人摔倒）

不依赖任何真实数据，完全由 ODE 动力学驱动。
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from backend.rl.dynamics import (
    step_plasma_state,
    PHYSICS_RANGES,
    compute_tau_E,
    compute_q95,
    compute_beta_N,
    denormalize_state,
    TAU_E_COEFFICIENTS,
)
from backend.rl.disruption import check_disruption
from backend.rl.rewards import compute_reward, compute_disruption_penalty


class FusionEnv(gym.Env):
    """
    托卡马克等离子体控制环境（Gymnasium API）

    观测空间（7维 Box [0,1]）：
        [n_e_norm, T_e_norm, B_norm, q95_norm, beta_N_norm, Ip_norm, P_heat_norm]

    动作空间（3维 Box [-0.1, 0.1]）：
        [delta_P_heat, delta_n_fuel, delta_Ip]

    最大步数：500步 = 5秒真实放电时间（dt=0.01s）
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, max_steps: int = 500, tau_e_coeffs: dict = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.max_steps   = max_steps
        self.tau_e_coeffs = tau_e_coeffs  # None → 使用 ITER98pY2 默认系数
        self.render_mode = render_mode

        # 状态空间：7维，归一化 [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # 动作空间：3维增量控制，[-0.1, 0.1]
        # 每步最多改变 10% 的归一化范围（物理上对应约 1-2 MW 功率变化）
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(3,), dtype=np.float32
        )

        self._state = None
        self._prev_state = None
        self._step_count = 0

        # 记录 episode 统计（供训练日志使用）
        self._episode_reward = 0.0
        self._disruption_occurred = False
        self._lawson_achieved = False

    def _sample_initial_state(self) -> np.ndarray:
        """
        随机采样初始状态（在物理合法范围内）

        初始状态需满足：
          - q95 > 2.5（远离 Kruskal-Shafranov 极限）
          - beta_N < 2.0（远离 Troyon 极限）
          - 密度在 Greenwald 极限的 30%~70%
        """
        rng = self.np_random

        # 随机采样各参数
        n_e_norm   = rng.uniform(0.2, 0.6)  # 不初始化在高密度区
        T_e_norm   = rng.uniform(0.1, 0.4)
        B_norm     = rng.uniform(0.55, 0.8)  # 最低 B=4.2T，失败 case 84%是 B<4.5T
        Ip_norm    = rng.uniform(0.2, 0.7)
        P_heat_norm = rng.uniform(0.1, 0.5)

        # 根据 Ip, B 反推 q95（归一化）
        Ip_lo, Ip_hi = PHYSICS_RANGES["Ip"]
        B_lo,  B_hi  = PHYSICS_RANGES["B"]
        q_lo,  q_hi  = PHYSICS_RANGES["q95"]
        bN_lo, bN_hi = PHYSICS_RANGES["beta_N"]
        n_lo,  n_hi  = PHYSICS_RANGES["n_e"]
        T_lo,  T_hi  = PHYSICS_RANGES["T_e"]

        Ip = Ip_lo + Ip_norm * (Ip_hi - Ip_lo)
        B  = B_lo  + B_norm  * (B_hi  - B_lo)
        n_e = n_lo + n_e_norm * (n_hi - n_lo)
        T_e = T_lo + T_e_norm * (T_hi - T_lo)

        q95   = compute_q95(Ip, B)
        beta_N = compute_beta_N(n_e, T_e, B, Ip)

        q95_norm   = np.clip((q95 - q_lo) / (q_hi - q_lo), 0.0, 1.0)
        beta_N_norm = np.clip((beta_N - bN_lo) / (bN_hi - bN_lo), 0.0, 1.0)

        state = np.array([
            n_e_norm, T_e_norm, B_norm,
            q95_norm, beta_N_norm,
            Ip_norm, P_heat_norm,
        ], dtype=np.float32)

        # 如果初始状态已破裂，重新采样（最多 10 次）
        for _ in range(10):
            disrupted, _ = check_disruption(state)
            if not disrupted:
                return state
            Ip_norm = rng.uniform(0.1, 0.5)  # 降低电流重试
            Ip = Ip_lo + Ip_norm * (Ip_hi - Ip_lo)
            q95 = compute_q95(Ip, B)
            q95_norm = np.clip((q95 - q_lo) / (q_hi - q_lo), 0.0, 1.0)
            state[3] = q95_norm
            state[5] = Ip_norm

        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._state         = self._sample_initial_state()
        self._prev_state    = self._state.copy()
        self._step_count    = 0
        self._episode_reward    = 0.0
        self._disruption_occurred = False
        self._lawson_achieved     = False

        obs  = self._state.copy()
        info = {"state_dict": denormalize_state(self._state)}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -0.1, 0.1).astype(np.float32)

        # ─── 物理演化 ────────────────────────────────────────────────────
        self._prev_state = self._state.copy()
        self._state = step_plasma_state(
            self._state, action, tau_e_coeffs=self.tau_e_coeffs
        )
        self._step_count += 1

        # ─── 检查破裂 ────────────────────────────────────────────────────
        disrupted, disruption_reason = check_disruption(self._state)

        # ─── 计算奖励 ────────────────────────────────────────────────────
        if disrupted:
            reward = compute_disruption_penalty()
            self._disruption_occurred = True
        else:
            reward, reward_info = compute_reward(
                self._state, action, self._prev_state, self.tau_e_coeffs
            )
            if reward_info["success_bonus"] > 0:
                self._lawson_achieved = True

        self._episode_reward += reward

        # ─── 终止条件 ────────────────────────────────────────────────────
        terminated = disrupted
        truncated  = self._step_count >= self.max_steps

        # ─── Info 字典 ───────────────────────────────────────────────────
        s_dict = denormalize_state(self._state)
        info = {
            "step":            self._step_count,
            "disrupted":       disrupted,
            "disruption_reason": disruption_reason if disrupted else "",
            "episode_reward":  self._episode_reward,
            "lawson_achieved": self._lawson_achieved,
            **s_dict,
        }
        if not disrupted:
            info.update(reward_info)

        obs = self._state.copy()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            s = denormalize_state(self._state)
            return (
                f"Step {self._step_count:4d} | "
                f"n_e={s['n_e']:.2e} | T_e={s['T_e']:.2e} | "
                f"q95={s['q95']:.3f} | beta_N={s['beta_N']:.3f} | "
                f"Ip={s['Ip']:.2e} | P_heat={s['P_heat']:.2e}"
            )

    def close(self):
        pass

    def get_lawson_parameter(self) -> float:
        """当前 Lawson 参数（用于 Dashboard 实时显示）"""
        from backend.rl.rewards import compute_lawson_parameter
        s = denormalize_state(self._state)
        tau_E = compute_tau_E(s["n_e"], s["B"], s["P_heat"], s["Ip"], self.tau_e_coeffs)
        return compute_lawson_parameter(s["n_e"], s["T_e"], tau_E)
