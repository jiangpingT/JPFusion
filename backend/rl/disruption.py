"""
disruption.py — 4个托卡马克破裂终止条件

类比 JPRobot 里的"摔倒检测"。
任何一条触发 → episode done，重置等离子体。
"""

import numpy as np
from backend.rl.dynamics import (
    PHYSICS_RANGES,
    compute_greenwald_density,
    compute_troyon_limit,
    denormalize_state,
)


def check_disruption(state_norm: np.ndarray) -> tuple[bool, str]:
    """
    检查是否发生等离子体破裂。

    Returns:
        (disrupted: bool, reason: str)
        reason 为空字符串表示未破裂
    """
    s = denormalize_state(state_norm)

    # ── 条件 1: q95 < 2.0 — Kruskal-Shafranov 不稳定 ─────────────────────
    # q95 < 2 时磁场线开始撕裂，MHD 不稳定性迅速增长
    if s["q95"] < 2.0:
        return True, "kruskal_shafranov: q95={:.3f} < 2.0".format(s["q95"])

    # ── 条件 2: beta_N > beta_N_limit — Troyon 稳定极限 ──────────────────
    # 等离子体压强相对磁压过大，触发理想 MHD 破裂（气球-撕裂模）
    beta_limit = compute_troyon_limit(s["Ip"], s["B"])
    if s["beta_N"] > beta_limit:
        return True, "troyon_limit: beta_N={:.3f} > {:.3f}".format(
            s["beta_N"], beta_limit)

    # ── 条件 3: n_e > n_Greenwald — Greenwald 密度极限 ───────────────────
    # 密度超过 Greenwald 极限时，辐射损失暴增，等离子体骤冷破裂
    n_G = compute_greenwald_density(s["Ip"])
    if s["n_e"] > n_G:
        return True, "greenwald_limit: n_e={:.2e} > n_G={:.2e}".format(
            s["n_e"], n_G)

    # ── 条件 4: 锁模不稳定性代理 — q95 * Ip 乘积异常 ─────────────────────
    # 锁模（Locked Mode）是 NTM（新古典撕裂模）锁定到误差场的结果
    # 简化代理：低 q95 + 高 Ip 同时出现 → 锁模风险高
    # 真实应用中需要 Mirnov 线圈信号，此处用解析代理
    locked_mode_proxy = (1.0 / max(s["q95"] - 2.0, 0.01)) * (s["Ip"] / 2e6)
    if locked_mode_proxy > 5.0:
        return True, "locked_mode_proxy={:.2f} > 5.0 (q95={:.3f}, Ip={:.2e})".format(
            locked_mode_proxy, s["q95"], s["Ip"])

    return False, ""


def disruption_margin(state_norm: np.ndarray) -> dict:
    """
    计算到各破裂边界的安全裕度（正数 = 安全，负数 = 已破裂）。
    用于奖励函数中的稳定性项。
    """
    s = denormalize_state(state_norm)
    beta_limit = compute_troyon_limit(s["Ip"], s["B"])
    n_G = compute_greenwald_density(s["Ip"])

    return {
        "q95_margin":    s["q95"] - 2.0,                    # 正数表示安全
        "beta_N_margin": beta_limit - s["beta_N"],           # 正数表示安全
        "density_margin": (n_G - s["n_e"]) / n_G,           # 相对裕度
        "locked_mode_safety": max(s["q95"] - 2.0, 0.01) * 2e6 / max(s["Ip"], 1e5),
    }
