"""
plasma_engine.py — PlasmaPy 封装，生成等离子体训练数据

物理量说明：
  n_e      — 电子密度 (m^-3)
  T_e      — 电子温度 (K，1eV ≈ 11600 K)
  B        — 磁场强度 (T，特斯拉)
  lambda_D — 德拜长度 (m)：等离子体中电场屏蔽距离
  omega_p  — 等离子体频率 (rad/s)：等离子体集体振荡频率
  v_A      — 阿尔文速度 (m/s)：磁流体中磁扰动传播速度
  beta     — β 值 (无量纲)：等离子体热压 / 磁压比
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple

# ─── 物理常数 ───────────────────────────────────────────────────────────────
EPSILON_0 = 8.854187817e-12   # 真空介电常数 (F/m)
K_B       = 1.380649e-23      # 玻尔兹曼常数 (J/K)
M_E       = 9.10938e-31       # 电子质量 (kg)
M_I       = 1.6726e-27        # 质子质量 (kg)，简化为氢等离子体
E_CHARGE  = 1.602176634e-19   # 元电荷 (C)
MU_0      = 1.25663706212e-6  # 真空磁导率 (H/m)
EV_TO_K   = 11604.52          # 1 eV 对应温度 (K)


def compute_debye_length(n_e: float, T_e: float) -> float:
    """
    德拜长度 λ_D = sqrt(ε₀ k_B T_e / n_e e²)
    参数：n_e (m^-3), T_e (K)
    返回：λ_D (m)
    """
    return np.sqrt(EPSILON_0 * K_B * T_e / (n_e * E_CHARGE**2))


def compute_plasma_frequency(n_e: float) -> float:
    """
    等离子体频率 ω_p = sqrt(n_e e² / ε₀ m_e)
    参数：n_e (m^-3)
    返回：ω_p (rad/s)
    """
    return np.sqrt(n_e * E_CHARGE**2 / (EPSILON_0 * M_E))


def compute_alfven_speed(n_e: float, B: float) -> float:
    """
    阿尔文速度 v_A = B / sqrt(μ₀ ρ)，ρ = n_e * m_i（质量密度）
    参数：n_e (m^-3), B (T)
    返回：v_A (m/s)
    """
    rho = n_e * M_I
    return B / np.sqrt(MU_0 * rho)


def compute_beta(n_e: float, T_e: float, B: float) -> float:
    """
    β = 2 μ₀ n_e k_B T_e / B²
    参数：n_e (m^-3), T_e (K), B (T)
    返回：β (无量纲)
    """
    return 2 * MU_0 * n_e * K_B * T_e / (B**2)


def compute_scalar_physics(n_e: float, T_e: float, B: float) -> Dict[str, float]:
    """
    计算所有标量物理量
    返回字典：{lambda_D, omega_p, v_alfven, beta}
    """
    return {
        "lambda_D":  compute_debye_length(n_e, T_e),
        "omega_p":   compute_plasma_frequency(n_e),
        "v_alfven":  compute_alfven_speed(n_e, B),
        "beta":      compute_beta(n_e, T_e, B),
    }


def add_mhd_perturbation(
    T_profile: np.ndarray,
    n_profile: np.ndarray,
    r_grid: np.ndarray,
    theta_grid: np.ndarray,
    beta: float,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加入 MHD（磁流体动力学）不稳定性扰动，使数据更真实

    物理背景：
      真实托卡马克中存在多种 MHD 不稳定模式（撕裂模、气球模等）
      β 值越高，等离子体压力越大，扰动越强烈
      角向模数 m、极向模数 n 决定扰动的空间形态

    参数：
      T_profile  — 温度 2D 分布数组
      n_profile  — 密度 2D 分布数组
      r_grid     — 半径坐标网格
      theta_grid — 角度坐标网格
      beta       — β 值（控制扰动强度）
      seed       — 随机种子（可复现）

    返回：(T_perturbed, n_perturbed) 扰动后的剖面，已截断到非负值
    """
    rng = np.random.RandomState(seed)
    # 扰动幅度随 β 增大，最大 15%
    epsilon = min(0.1 * beta, 0.15)

    T_out = T_profile.copy()
    n_out = n_profile.copy()

    # 叠加多个 MHD 不稳定性模式（角向模数 m，极向模数 n）
    m_modes = [1, 2, 3]  # 角向模数（典型 MHD 不稳定性模式）
    n_modes = [1, 2]     # 极向模数

    for m in m_modes:
        for n in n_modes:
            amp   = epsilon * rng.uniform(0, 1) / (m * n)
            phase = rng.uniform(0, 2 * np.pi)
            pattern = np.sin(m * theta_grid + n * r_grid * np.pi + phase)
            # 温度场受全幅扰动，密度场扰动幅度减半（密度约束比温度更强）
            T_out += amp * T_out * pattern
            n_out += amp * 0.5 * n_out * pattern

    return T_out.clip(0), n_out.clip(0)


def generate_plasma_profile_2d(
    n_e: float,
    T_e: float,
    B: float,
    grid_size: int = 32,
    add_turbulence: bool = False,
) -> Dict[str, np.ndarray]:
    """
    生成 2D 等离子体截面分布（托卡马克截面极坐标高斯模型）

    物理假设：
      - 温度剖面：中心峰化高斯分布，T(r) = T_e * exp(-r² / 2σ²)
      - 密度剖面：中心峰化，n(r) = n_e * exp(-r² / 2σ_n²)
      - σ_T = 0.3（温度宽度），σ_n = 0.25（密度宽度），r ∈ [0, 1]
      - 托卡马克约束：r > 0.95 的边界区域温度骤降（刮削层效应）
      - add_turbulence=True 时叠加 MHD 不稳定性扰动

    参数：
      n_e, T_e, B    — 等离子体参数
      grid_size      — 网格分辨率（grid_size × grid_size）
      add_turbulence — 是否叠加 MHD 湍流扰动

    返回：
      {
        "r_grid":   r 坐标矩阵 (grid_size, grid_size)
        "theta_grid": θ 坐标矩阵
        "T_profile": 温度分布 (归一化)
        "n_profile": 密度分布 (归一化)
        "x_grid":   笛卡尔 x 坐标
        "y_grid":   笛卡尔 y 坐标
      }
    """
    # 极坐标网格，r ∈ [0, 1]，θ ∈ [0, 2π]
    r_lin     = np.linspace(0, 1, grid_size)
    theta_lin = np.linspace(0, 2 * np.pi, grid_size)
    r_grid, theta_grid = np.meshgrid(r_lin, theta_lin)

    # 高斯温度剖面
    sigma_T = 0.30
    T_profile = T_e * np.exp(-r_grid**2 / (2 * sigma_T**2))

    # 边界效应：刮削层（SOL）指数衰减模型
    # 物理依据：SOL 区域热流以特征长度 lambda_q 指数衰减
    # T_SOL(r) = T_LCFS * exp(-(r - r_sep) / lambda_q)
    # r_sep=0.90 为最后封闭磁通面（LCFS），lambda_q=0.03 为 SOL 特征衰减长度
    r_sep    = 0.90
    lambda_q = 0.03
    T_lcfs   = T_e * np.exp(-r_sep**2 / (2 * sigma_T**2))   # 分界面处温度
    sol_mask = r_grid > r_sep
    T_profile[sol_mask] = T_lcfs * np.exp(-(r_grid[sol_mask] - r_sep) / lambda_q)

    # 高斯密度剖面
    sigma_n   = 0.35
    n_profile = n_e * np.exp(-r_grid**2 / (2 * sigma_n**2))

    # 磁场对密度约束的影响（β 越大，约束越差，密度分布展宽）
    beta_val  = compute_beta(n_e, T_e, B)
    spread    = 1 + 0.2 * min(beta_val, 1.0)
    n_profile = n_e * np.exp(-r_grid**2 / (2 * (sigma_n * spread)**2))

    # MHD 湍流扰动
    if add_turbulence:
        T_profile, n_profile = add_mhd_perturbation(
            T_profile, n_profile, r_grid, theta_grid, beta_val
        )

    # 笛卡尔坐标（用于可视化）
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    return {
        "r_grid":     r_grid.tolist(),
        "theta_grid": theta_grid.tolist(),
        "T_profile":  T_profile.tolist(),
        "n_profile":  n_profile.tolist(),
        "x_grid":     x_grid.tolist(),
        "y_grid":     y_grid.tolist(),
    }


def generate_training_dataset(
    n_samples: int = 5000,
    grid_size: int = 16,
    save_path: str = None,
    add_turbulence: bool = True,
) -> List[Dict]:
    """
    生成训练数据集：参数扫描 + 逐点采样

    策略：
      - 在参数空间 (n_e, T_e, B) 随机采样
      - 对每组参数，在截面上采样 grid_size² 个空间点
      - 每个样本：(n_e, T_e, B, r, theta) → T_at_point
      - add_turbulence=True 时叠加 MHD 湍流扰动，使数据更真实

    参数范围（聚变相关量级）：
      n_e: 1e18 ~ 1e20 (m^-3)   — 托卡马克典型密度
      T_e: 1e6  ~ 1e8  (K)      — 对应 100eV ~ 10keV
      B:   1    ~ 10   (T)      — 典型托卡马克磁场

    返回：每个样本的字典列表
    """
    np.random.seed(42)

    # 参数空间（对数均匀采样）
    n_e_samples = np.power(10, np.random.uniform(18, 20, n_samples))
    T_e_samples = np.power(10, np.random.uniform(6,  8,  n_samples))
    B_samples   = np.random.uniform(1, 10, n_samples)

    # 空间坐标采样（建立网格）
    r_values     = np.linspace(0, 1, grid_size)
    theta_values = np.linspace(0, 2 * np.pi, grid_size)
    r_grid, theta_grid = np.meshgrid(r_values, theta_values)
    # r_grid shape: (grid_size, grid_size)，r_grid[ti, ri] = r_values[ri]

    dataset = []
    turb_str = "（含 MHD 湍流扰动）" if add_turbulence else ""
    print(f"[PlasmaPy] 开始生成 {n_samples} 组参数 × {grid_size}² 空间点 = {n_samples * grid_size**2} 样本...{turb_str}")

    for i in range(n_samples):
        n_e = n_e_samples[i]
        T_e = T_e_samples[i]
        B   = B_samples[i]

        # 计算物理量
        physics  = compute_scalar_physics(n_e, T_e, B)
        beta_val = physics["beta"]
        sigma_T  = 0.30
        sigma_n  = 0.35
        spread   = 1 + 0.2 * min(beta_val, 1.0)

        # 生成 2D 剖面（矢量化，然后对整块应用扰动）
        T_2d = T_e * np.exp(-r_grid**2 / (2 * sigma_T**2))
        # SOL 指数衰减边界
        r_sep    = 0.90
        lambda_q = 0.03
        T_lcfs   = T_e * np.exp(-r_sep**2 / (2 * sigma_T**2))
        sol_mask = r_grid > r_sep
        T_2d[sol_mask] = T_lcfs * np.exp(-(r_grid[sol_mask] - r_sep) / lambda_q)

        n_2d = n_e * np.exp(-r_grid**2 / (2 * (sigma_n * spread)**2))

        # MHD 湍流扰动（seed=i 保证每组参数扰动不同但可复现）
        if add_turbulence:
            T_2d, n_2d = add_mhd_perturbation(
                T_2d, n_2d, r_grid, theta_grid, beta_val, seed=i
            )

        # 从 2D 剖面采样所有空间点
        for ti, theta in enumerate(theta_values):
            for ri, r in enumerate(r_values):
                T_at_point = float(T_2d[ti, ri])
                n_at_point = float(n_2d[ti, ri])

                dataset.append({
                    # 输入特征（归一化在 dataset.py 里做）
                    "n_e":   float(n_e),
                    "T_e":   float(T_e),
                    "B":     float(B),
                    "r":     float(r),
                    "theta": float(theta),
                    # 输出目标
                    "T_out": T_at_point,
                    "n_out": n_at_point,
                    # 辅助物理量（用于验证）
                    "lambda_D": float(physics["lambda_D"]),
                    "beta":     float(physics["beta"]),
                })

        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{n_samples} 组参数")

    print(f"[PlasmaPy] 数据集生成完成，共 {len(dataset)} 条样本")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(dataset[:50000], f)  # 限制文件大小
        print(f"[PlasmaPy] 已保存到 {save_path}")

    return dataset


# ─── 独立测试入口 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 测试标量计算
    n_e_test = 1e19   # 典型托卡马克密度
    T_e_test = 1e7    # 约 860 eV
    B_test   = 5.0    # 5 特斯拉

    phys = compute_scalar_physics(n_e_test, T_e_test, B_test)
    print("=== 标量物理量测试 ===")
    print(f"  德拜长度 λ_D  = {phys['lambda_D']:.4e} m")
    print(f"  等离子体频率  = {phys['omega_p']:.4e} rad/s")
    print(f"  阿尔文速度    = {phys['v_alfven']:.4e} m/s")
    print(f"  β 值          = {phys['beta']:.4e}")

    # 测试 2D 分布（含湍流）
    profile = generate_plasma_profile_2d(n_e_test, T_e_test, B_test, grid_size=8, add_turbulence=True)
    print("\n=== 2D 剖面测试（含 MHD 扰动）===")
    print(f"  T_profile shape: {np.array(profile['T_profile']).shape}")
    print(f"  T_max = {np.array(profile['T_profile']).max():.3e}")
    print(f"  T_min = {np.array(profile['T_profile']).min():.3e}")
