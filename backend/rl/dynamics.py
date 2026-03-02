"""
dynamics.py — 托卡马克等离子体 ODE 动力学模型

用简化 1st-order ODE 近似托卡马克等离子体的时间演化。
tau_E 用 ITER98pY2 经验公式（可被 Phase 2A 数据校准覆盖）。

控制周期 dt = 0.01s
"""

import numpy as np

# ─── ITER98pY2 能量约束时间 τ_E 经验公式系数 ─────────────────────────────
# τ_E = C * H98 * Ip^0.93 * B^0.15 * P^-0.69 * n^0.41 * M^0.19 * R^1.97 * ε^0.58 * κ^0.78
# 简化版（固定 EAST 几何参数）：τ_E ∝ n^a * T^b * B^c * P^d
# 系数可被 Phase 2A fit_tau_e_coefficients() 更新

TAU_E_COEFFICIENTS = {
    "C":   0.0562,   # 归一化常数（ITER98pY2 标准值）
    "a":   0.41,     # 密度指数
    "b":  -0.00,     # 温度指数（τ_E 不直接依赖 T_e）
    "c":   0.15,     # 磁场指数
    "d":  -0.69,     # 加热功率指数
    "e":   0.93,     # 等离子体电流指数
    # EAST 固定几何参数
    "R":   1.85,     # 大半径 (m)
    "a_minor": 0.45, # 小半径 (m)
    "kappa":   1.7,  # 拉伸比
    "M":   2.0,      # 氘粒子质量数
}

# ─── 物理范围（归一化前的真实量纲单位）─────────────────────────────────────
PHYSICS_RANGES = {
    "n_e":    (1e19, 1e20),     # 电子密度 (m^-3)
    "T_e":    (1e6,  5e7),      # 电子温度 (K)，对应 0.1keV ~ 5keV
    "B":      (2.0,  6.0),      # 磁场强度 (T)
    "q95":    (2.0,  8.0),      # 安全因子（>2 才稳定）
    "beta_N": (0.0,  4.0),      # 归一化 beta（Troyon 极限约 3.5）
    "Ip":     (0.5e6, 2.0e6),   # 等离子体电流 (A)
    "P_heat": (0.5e6, 20e6),    # 加热功率 (W)
}

DT = 0.01  # 控制周期 (s)


def compute_tau_E(n_e: float, B: float, P_heat: float, Ip: float,
                  coeffs: dict = None) -> float:
    """
    计算能量约束时间 τ_E（ITER98pY2 简化版）

    参数均为真实量纲值（非归一化）
    Returns: τ_E (s)
    """
    c = coeffs if coeffs is not None else TAU_E_COEFFICIENTS
    n_e_19 = n_e / 1e19        # 单位：10^19 m^-3
    P_MW   = max(P_heat / 1e6, 0.1)  # 单位：MW，避免除零
    Ip_MA  = max(Ip / 1e6, 0.01)     # 单位：MA

    tau_E = (c["C"]
             * (n_e_19 ** c["a"])
             * (B ** c["c"])
             * (P_MW ** c["d"])
             * (Ip_MA ** c["e"])
             * (c["R"] ** 1.97)
             * (c["a_minor"] ** 0.58)
             * (c["kappa"] ** 0.78)
             * (c["M"] ** 0.19))
    return float(np.clip(tau_E, 0.001, 5.0))  # 夹到 1ms ~ 5s


def compute_greenwald_density(Ip: float) -> float:
    """
    Greenwald 密度极限: n_G = Ip / (π * a²) × 10^20
    Ip 单位：A，a 单位：m
    Returns: n_G (m^-3)
    """
    a = TAU_E_COEFFICIENTS["a_minor"]
    Ip_MA = Ip / 1e6
    return Ip_MA / (np.pi * a ** 2) * 1e20


def compute_troyon_limit(Ip: float, B: float) -> float:
    """
    Troyon β_N 稳定极限（典型值 3.5，实际与 κ/q 有关）
    beta_N_limit * Ip / (a * B) = β
    Returns: beta_N_limit（无量纲）
    """
    return 3.5  # 简化为常数


def compute_q95(Ip: float, B: float) -> float:
    """
    q95 安全因子近似（圆截面简化公式）
    q = 5 * a² * B / (R * Ip) × 1e6
    """
    a = TAU_E_COEFFICIENTS["a_minor"]
    R = TAU_E_COEFFICIENTS["R"]
    return 5.0 * a ** 2 * B * 1e6 / (R * Ip)


def compute_beta_N(n_e: float, T_e: float, B: float, Ip: float) -> float:
    """
    归一化 beta（beta_N = β × a × B / Ip）
    β = n_e * k_B * T_e / (B²/2μ₀)
    k_B = 1.38e-23 J/K，μ₀ = 4π×10^-7
    """
    k_B = 1.38e-23
    mu_0 = 4 * np.pi * 1e-7
    a = TAU_E_COEFFICIENTS["a_minor"]
    beta = (n_e * k_B * T_e) / (B ** 2 / (2 * mu_0))
    Ip_MA = max(Ip / 1e6, 0.01)
    return beta * a * B / Ip_MA


def step_plasma_state(state: np.ndarray, action: np.ndarray, dt: float = DT,
                      tau_e_coeffs: dict = None) -> np.ndarray:
    """
    等离子体状态一步 ODE 积分（RK4 法，O(dt⁵) 误差，优于 Euler 的 O(dt²)）

    State（7维，归一化 0~1）：
        [n_e_norm, T_e_norm, B_norm, q95_norm, beta_N_norm, Ip_norm, P_heat_norm]

    Action（3维，增量）：
        [delta_P_heat, delta_n_fuel, delta_Ip]
        每维范围 -0.1 ~ +0.1（归一化单位）

    Returns: next_state（7维，归一化，夹到合法范围）
    """
    n_e_norm, T_e_norm, B_norm, q95_norm, beta_N_norm, Ip_norm, P_heat_norm = state
    delta_P, delta_n, delta_Ip = action

    # ─── 反归一化到真实量纲 ──────────────────────────────────────────────
    n_lo, n_hi   = PHYSICS_RANGES["n_e"]
    T_lo, T_hi   = PHYSICS_RANGES["T_e"]
    B_lo, B_hi   = PHYSICS_RANGES["B"]
    Ip_lo, Ip_hi = PHYSICS_RANGES["Ip"]
    P_lo, P_hi   = PHYSICS_RANGES["P_heat"]

    n_e   = n_lo + n_e_norm * (n_hi - n_lo)
    T_e   = T_lo + T_e_norm * (T_hi - T_lo)
    B     = B_lo + B_norm * (B_hi - B_lo)
    Ip    = Ip_lo + Ip_norm * (Ip_hi - Ip_lo)
    P_heat = P_lo + P_heat_norm * (P_hi - P_lo)

    # ─── 应用动作（增量控制）────────────────────────────────────────────
    P_heat_next = np.clip(P_heat + delta_P * (P_hi - P_lo), P_lo, P_hi)
    n_e_next    = np.clip(n_e + delta_n * (n_hi - n_lo), n_lo, n_hi)
    Ip_next     = np.clip(Ip + delta_Ip * (Ip_hi - Ip_lo), Ip_lo, Ip_hi)

    # ─── 物理响应：温度由能量平衡方程决定 ────────────────────────────────
    # dT/dt = (P_heat - P_rad) / (3/2 * n_e * k_B * V) / tau_E
    # 简化：T_e 向平衡温度弛豫（时间常数 = tau_E）
    tau_E = compute_tau_E(n_e_next, B, P_heat_next, Ip_next, tau_e_coeffs)

    # 平衡温度估算：W = P_heat * tau_E = 3/2 * n_e * k_B * T_eq * V
    # V ≈ 2π²Ra² ≈ 2π²×1.85×0.45² ≈ 7.45 m³（EAST 典型值）
    V_plasma = 2 * np.pi ** 2 * TAU_E_COEFFICIENTS["R"] * TAU_E_COEFFICIENTS["a_minor"] ** 2
    k_B = 1.38e-23
    T_eq = (P_heat_next * tau_E) / (1.5 * n_e_next * k_B * V_plasma)
    T_eq = np.clip(T_eq, T_lo, T_hi)

    # 1st-order 弛豫 ODE：T_e 向 T_eq 靠拢，时间常数 = tau_E（RK4 积分）
    dT = lambda T: (T_eq - T) / tau_E
    k1 = dt * dT(T_e)
    k2 = dt * dT(T_e + k1 / 2)
    k3 = dt * dT(T_e + k2 / 2)
    k4 = dt * dT(T_e + k3)
    T_e_next = T_e + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    T_e_next = np.clip(T_e_next, T_lo, T_hi)

    # B 本次不可控（固定在设定值 + 小幅感应效应）
    B_next = B  # 简化：磁场恒定（由外部线圈决定）

    # ─── 导出量：q95, beta_N ──────────────────────────────────────────────
    q95_next   = compute_q95(Ip_next, B_next)
    beta_N_next = compute_beta_N(n_e_next, T_e_next, B_next, Ip_next)

    # ─── 重新归一化 ──────────────────────────────────────────────────────
    q_lo, q_hi   = PHYSICS_RANGES["q95"]
    bN_lo, bN_hi = PHYSICS_RANGES["beta_N"]

    next_state = np.array([
        (n_e_next   - n_lo)   / (n_hi - n_lo),
        (T_e_next   - T_lo)   / (T_hi - T_lo),
        (B_next     - B_lo)   / (B_hi - B_lo),
        (q95_next   - q_lo)   / (q_hi - q_lo),
        (beta_N_next - bN_lo) / (bN_hi - bN_lo),
        (Ip_next    - Ip_lo)  / (Ip_hi - Ip_lo),
        (P_heat_next - P_lo)  / (P_hi - P_lo),
    ], dtype=np.float32)

    return np.clip(next_state, 0.0, 1.0)


def denormalize_state(state_norm: np.ndarray) -> dict:
    """将归一化状态向量还原为真实量纲值，返回命名字典"""
    n_lo, n_hi   = PHYSICS_RANGES["n_e"]
    T_lo, T_hi   = PHYSICS_RANGES["T_e"]
    B_lo, B_hi   = PHYSICS_RANGES["B"]
    q_lo, q_hi   = PHYSICS_RANGES["q95"]
    bN_lo, bN_hi = PHYSICS_RANGES["beta_N"]
    Ip_lo, Ip_hi = PHYSICS_RANGES["Ip"]
    P_lo, P_hi   = PHYSICS_RANGES["P_heat"]

    s = state_norm
    return {
        "n_e":    float(n_lo + s[0] * (n_hi - n_lo)),
        "T_e":    float(T_lo + s[1] * (T_hi - T_lo)),
        "B":      float(B_lo + s[2] * (B_hi - B_lo)),
        "q95":    float(q_lo + s[3] * (q_hi - q_lo)),
        "beta_N": float(bN_lo + s[4] * (bN_hi - bN_lo)),
        "Ip":     float(Ip_lo + s[5] * (Ip_hi - Ip_lo)),
        "P_heat": float(P_lo + s[6] * (P_hi - P_lo)),
    }
