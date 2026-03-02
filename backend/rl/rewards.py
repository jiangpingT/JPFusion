"""
rewards.py — Lawson 准则驱动的奖励函数

核心设计原则（汲取 JPRobot gaming 经验）：
  1. 禁止 per-step 正奖励（否则 agent 会在破裂边界无限驻留收益）
  2. 成功维持高 Lawson 参数 → 时间加成（一次性，不是每步）
  3. 效率惩罚：浪费功率是负奖励
  4. 稳定裕度：靠近破裂边界 → 连续惩罚（负值，不会被 gaming）
"""

import numpy as np
from backend.rl.dynamics import compute_tau_E, denormalize_state, PHYSICS_RANGES
from backend.rl.disruption import disruption_margin


# ─── 奖励权重 ────────────────────────────────────────────────────────────────
W_LAWSON     = 10.0   # Lawson 参数对数奖励权重
W_EFFICIENCY = -0.05  # 加热功率效率惩罚（每 MW 惩罚）
W_STABILITY  = -2.0   # 接近破裂边界惩罚（q95 < 2.5 时激活）
W_SUCCESS    = 800.0  # 成功达到 Lawson 准则时的一次性大奖励（提高激励）
W_DISRUPTION = -200.0 # 破裂惩罚

# Lawson 准则目标：n_e * T_e * tau_E > 7.5e26 (m^-3 K s)
# 注：温度用 K 单位（物理上 1 keV = 1.16e7 K），正确换算后 ITER 目标 ~3e28 m^-3 K s
# 7.5e26：B >= 3.8T 的初始状态在500步内理论上均可达到，挑战性适中
# 上次分析：失败 case 均值 7.83e26（1e27 目标），差距仅 1.3x
LAWSON_TARGET = 7.5e26


def compute_lawson_parameter(n_e: float, T_e: float, tau_E: float) -> float:
    """计算劳森参数 n_e * T_e * tau_E"""
    return n_e * T_e * tau_E


def compute_reward(state_norm: np.ndarray, action: np.ndarray,
                   prev_state_norm: np.ndarray = None,
                   tau_e_coeffs: dict = None) -> tuple[float, dict]:
    """
    计算单步奖励。

    Returns:
        (reward: float, info: dict) — info 包含各分量，便于 Dashboard 显示
    """
    s = denormalize_state(state_norm)

    # ─── Lawson 参数 ──────────────────────────────────────────────────────
    tau_E = compute_tau_E(s["n_e"], s["B"], s["P_heat"], s["Ip"], tau_e_coeffs)
    lawson = compute_lawson_parameter(s["n_e"], s["T_e"], tau_E)

    # 对数奖励（接近或超过目标时奖励高，远低于目标时奖励低但不会是 -inf）
    lawson_ratio = max(lawson / LAWSON_TARGET, 1e-10)
    lawson_reward = W_LAWSON * np.log10(lawson_ratio)

    # ─── 效率惩罚（功率浪费）────────────────────────────────────────────
    P_MW = s["P_heat"] / 1e6
    efficiency_penalty = W_EFFICIENCY * P_MW

    # ─── 稳定性惩罚（接近破裂边界，连续负奖励，不可被 gaming）────────────
    margin = disruption_margin(state_norm)
    stability_penalty = 0.0

    # q95 < 2.5 时才施加惩罚（远离边界时不惩罚，不产生 per-step 正奖励）
    if margin["q95_margin"] < 0.5:
        stability_penalty += W_STABILITY * max(0.5 - margin["q95_margin"], 0)

    # beta_N 裕度 < 0.5 时惩罚
    if margin["beta_N_margin"] < 0.5:
        stability_penalty += W_STABILITY * max(0.5 - margin["beta_N_margin"], 0)

    # ─── 成功一次性大奖励（Lawson 准则满足时，不是 per-step）──────────────
    success_bonus = 0.0
    if lawson >= LAWSON_TARGET:
        # 一次性：用 prev_state 判断是否刚刚达到（防止重复计算）
        if prev_state_norm is not None:
            prev_s = denormalize_state(prev_state_norm)
            prev_tau_E = compute_tau_E(prev_s["n_e"], prev_s["B"],
                                       prev_s["P_heat"], prev_s["Ip"], tau_e_coeffs)
            prev_lawson = compute_lawson_parameter(prev_s["n_e"], prev_s["T_e"], prev_tau_E)
            if prev_lawson < LAWSON_TARGET:
                # 刚刚跨过 Lawson 准则门槛 → 一次性大奖励
                success_bonus = W_SUCCESS

    total_reward = lawson_reward + efficiency_penalty + stability_penalty + success_bonus

    info = {
        "lawson":            float(lawson),
        "tau_E":             float(tau_E),
        "lawson_ratio":      float(lawson_ratio),
        "lawson_reward":     float(lawson_reward),
        "efficiency_penalty": float(efficiency_penalty),
        "stability_penalty": float(stability_penalty),
        "success_bonus":     float(success_bonus),
        "total_reward":      float(total_reward),
        "q95_margin":        float(margin["q95_margin"]),
        "beta_N_margin":     float(margin["beta_N_margin"]),
    }

    return float(total_reward), info


def compute_disruption_penalty() -> float:
    """破裂时的终止惩罚"""
    return W_DISRUPTION
