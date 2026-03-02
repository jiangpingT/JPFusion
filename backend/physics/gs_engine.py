"""
gs_engine.py — FreeGS Grad-Shafranov 平衡求解器封装

用途：
  取代简化的高斯径向剖面，用磁通面坐标 ψ_N(R,Z) 重建温度/压强分布。
  物理上更正确：温度在磁通面上均匀，由输运决定的 ψ_N 剖面决定各面上的值。

物理流程：
  1. 从 (n_e, T_e, B) 推导 FreeGS 输入参数（轴压、等离子体电流、f_vac）
  2. 求解 Grad-Shafranov 方程：Δ*ψ = -μ₀R²dp/dψ - F dF/dψ
  3. 提取归一化磁通 ψ_N(R,Z)：0=磁轴，1=LCFS（最后封闭磁通面）
  4. 温度剖面：T(ψ_N) = T_axis * (1 - ψ_N^α_m)^α_n（α profile）
  5. SOL 区域（ψ_N > 1）：基于 psi 偏差的指数衰减

依赖：freegs（pip install freegs）

典型用时：nx=ny=33 网格约 1-3 秒
"""

import warnings
import numpy as np
from typing import Dict, Optional, Tuple

# ─── 物理常数 ──────────────────────────────────────────────────────────────
K_B  = 1.380649e-23       # 玻尔兹曼常数 (J/K)
MU_0 = 1.25663706212e-6   # 真空磁导率 (H/m)

# ─── 机器几何（简化中型托卡马克，类 JET/EAST 参数）──────────────────────
R0   = 1.50   # 大半径 (m)
a    = 0.50   # 小半径 (m)
RMIN = R0 - a           # 1.00 m
RMAX = R0 + a           # 2.00 m
ZMIN = -(a * 1.5)       # -0.75 m
ZMAX =  (a * 1.5)       #  0.75 m


def _ne_Te_B_to_gs_params(n_e: float, T_e: float, B: float) -> Dict[str, float]:
    """
    将等离子体参数 (n_e, T_e, B) 映射到 FreeGS 输入参数

    物理映射：
      p_axis = 2 n_e k_B T_e     — 轴上总压强（电子+离子，T_i≈T_e）[Pa]
      I_p    = 2π a² B/(μ₀ R₀ q₉₅)  — Wesson q₉₅=3.0 估算等离子体电流 [A]
      f_vac  = R₀ · B             — 真空中环向磁通函数 f=R·Bφ [T·m]
    """
    p_axis = 2.0 * n_e * K_B * T_e
    q95    = 3.0
    I_p    = 2 * np.pi * a**2 * B / (MU_0 * R0 * q95)
    f_vac  = R0 * B
    return {"p_axis": p_axis, "I_p": I_p, "f_vac": f_vac}


def solve_gs_equilibrium(
    n_e:     float,
    T_e:     float,
    B:       float,
    nx:      int   = 33,
    ny:      int   = 33,
    alpha_m: float = 1.0,
    alpha_n: float = 2.0,
    maxits:  int   = 25,
) -> Tuple[object, Dict]:
    """
    求解 Grad-Shafranov 平衡方程，返回平衡对象和关键元数据

    参数：
      n_e, T_e, B  — 等离子体参数（密度 m^-3、温度 K、磁场 T）
      nx, ny       — 求解网格分辨率（必须为 2^n+1，如 17/33/65）
      alpha_m/n    — 电流剖面峰化参数（控制 J_tor 的形状）
      maxits       — Picard 迭代最大次数

    返回：
      (eq, meta)
      eq   — freegs.Equilibrium 对象
      meta — 关键标量字典（psi_axis, psi_bndry, I_p, poloidalBeta, q95...）
    """
    try:
        import freegs
        from freegs import jtor, boundary
    except ImportError:
        raise RuntimeError("FreeGS 未安装，请运行: pip install freegs")

    gs = _ne_Te_B_to_gs_params(n_e, T_e, B)

    # ── 固定边界平衡（无需线圈系统，快速、稳定）────────────────────────────
    eq = freegs.Equilibrium(
        Rmin=RMIN, Rmax=RMAX,
        Zmin=ZMIN, Zmax=ZMAX,
        nx=nx, ny=ny,
        boundary=boundary.fixedBoundary,
    )

    profiles = jtor.ConstrainPaxisIp(
        eq,
        paxis   = gs["p_axis"],
        Ip      = gs["I_p"],
        fvac    = gs["f_vac"],
        alpha_m = alpha_m,
        alpha_n = alpha_n,
    )

    # Picard 迭代求解（抑制积分警告）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freegs.solve(eq, profiles, maxits=maxits)

    # ── 提取元数据 ─────────────────────────────────────────────────────────
    meta = {
        "p_axis":         float(gs["p_axis"]),
        "I_p_input":      float(gs["I_p"]),
        "plasma_current": float(eq.plasmaCurrent()),
        "poloidal_beta":  float(eq.poloidalBeta()),
        "psi_axis":       float(eq.psi_axis),
        "psi_bndry":      float(eq.psi_bndry),
        "R0":             R0,
        "a":              a,
        "alpha_m":        alpha_m,
        "alpha_n":        alpha_n,
    }

    # q-profile（fixedBoundary 无 X-point，捕获异常）
    try:
        psinorm_grid = np.linspace(0.1, 0.9, 9)
        q_vals = eq.q(psinorm_grid)
        meta["q_axis"] = float(q_vals[0])
        meta["q95"]    = float(q_vals[-1])
    except Exception:
        meta["q_axis"] = None
        meta["q95"]    = None

    return eq, meta


def build_temperature_on_grid(
    eq,
    T_e:       float,
    grid_size: int   = 32,
    alpha_m:   float = 1.0,
    alpha_n:   float = 2.0,
    lambda_q:  float = 0.04,
) -> Dict:
    """
    在极坐标输出网格上，利用磁通面坐标 ψ_N 重建温度剖面

    温度模型：
      CORE (ψ_N ≤ 1): T(ψ_N) = T_axis · (1 - ψ_N^α_m)^α_n  [α 剖面]
      SOL  (ψ_N > 1): T = T_LCFS · exp(-Δψ_N / λ_q)           [指数衰减]

    参数：
      eq        — solve_gs_equilibrium 返回的平衡对象
      T_e       — 输入电子温度（用于定标轴上温度）[K]
      grid_size — 输出网格分辨率（grid_size × grid_size，极坐标）
      alpha_m/n — 剖面峰化参数（同求解时使用的值）
      lambda_q  — SOL ψ_N 衰减尺度（无量纲，类比真实 λ_q/a）

    返回字典含：
      T_profile    — 温度分布 [K]（grid_size × grid_size，list of lists）
      psiN_profile — 归一化磁通分布（0=轴，1=LCFS）
      r_grid       — 归一化小半径坐标
      theta_grid   — 极角坐标
      x_grid, y_grid — 笛卡尔（归一化）坐标（用于可视化）
    """
    from scipy.interpolate import RegularGridInterpolator

    # ── 极坐标输出网格 → (R, Z) 真实坐标 ───────────────────────────────────
    r_norm     = np.linspace(0, 1, grid_size)            # 归一化小半径
    theta_vals = np.linspace(0, 2 * np.pi, grid_size)
    r_grid, theta_grid = np.meshgrid(r_norm, theta_vals)  # shape (gs, gs)

    # 圆形截面近似：R = R0 + a·r·cos θ，Z = a·r·sin θ
    R_pts = R0 + a * r_grid * np.cos(theta_grid)
    Z_pts =      a * r_grid * np.sin(theta_grid)

    # ── 从 FreeGS 网格插值 ψ_N ────────────────────────────────────────────
    R_gs   = eq.R[:, 0]   # (nx,) — FreeGS 径向坐标
    Z_gs   = eq.Z[0, :]   # (ny,) — FreeGS 轴向坐标
    psiN_2d = eq.psiN()   # (nx, ny)

    interp_psiN = RegularGridInterpolator(
        (R_gs, Z_gs), psiN_2d,
        method="linear",
        bounds_error=False,
        fill_value=1.5,   # 域外默认 1.5（深 SOL）
    )

    pts      = np.stack([R_pts.flatten(), Z_pts.flatten()], axis=-1)
    psiN_flat = interp_psiN(pts)
    psiN_grid = psiN_flat.reshape(grid_size, grid_size)

    # ── 温度剖面映射 ─────────────────────────────────────────────────────
    T_axis = T_e                       # 轴上温度 = 输入 T_e
    T_lcfs = 0.05 * T_axis             # LCFS 处约为轴上 5%（典型 H-mode）

    T_profile = np.zeros_like(psiN_grid)

    # CORE: α profile  T(ψ_N) = T_axis * (1 - ψ_N^α_m)^α_n
    core_mask = psiN_grid <= 1.0
    psiN_c    = np.clip(psiN_grid[core_mask], 0.0, 1.0)
    T_profile[core_mask] = T_axis * (1.0 - psiN_c ** alpha_m) ** alpha_n

    # SOL: 指数衰减（Δψ_N 越大温度越低）
    sol_mask  = psiN_grid > 1.0
    delta_psiN = psiN_grid[sol_mask] - 1.0
    T_profile[sol_mask] = T_lcfs * np.exp(-delta_psiN / lambda_q)

    T_profile = np.clip(T_profile, 0.0, None)

    # ── 笛卡尔归一化坐标（可视化用）─────────────────────────────────────
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    return {
        "T_profile":    T_profile.tolist(),
        "psiN_profile": psiN_grid.tolist(),
        "r_grid":       r_grid.tolist(),
        "theta_grid":   theta_grid.tolist(),
        "x_grid":       x_grid.tolist(),
        "y_grid":       y_grid.tolist(),
    }


def generate_plasma_profile_2d_gs(
    n_e:       float,
    T_e:       float,
    B:         float,
    grid_size: int   = 32,
    alpha_m:   float = 1.0,
    alpha_n:   float = 2.0,
) -> Dict:
    """
    完整流程：(n_e, T_e, B) → GS 平衡 → 温度/psiN 二维分布

    这是 plasma_engine.py 中 generate_plasma_profile_2d() 的物理增强版本。
    输出字段与原接口兼容，额外新增：psiN_profile, gs_meta

    返回：
      {
        "T_profile":    [[...]] — 温度 [K]（grid_size × grid_size）
        "psiN_profile": [[...]] — 归一化磁通（0=轴，1=LCFS）
        "r_grid", "theta_grid", "x_grid", "y_grid" — 坐标
        "n_profile":   [[...]]  — 密度剖面（基于 ψ_N，与温度同构）
        "gs_meta":     {...}    — 平衡关键参数
      }
    """
    # 求解 G-S 平衡
    eq, meta = solve_gs_equilibrium(
        n_e=n_e, T_e=T_e, B=B,
        nx=33, ny=33,
        alpha_m=alpha_m, alpha_n=alpha_n,
    )

    # 温度剖面
    grid_data = build_temperature_on_grid(
        eq, T_e=T_e, grid_size=grid_size,
        alpha_m=alpha_m, alpha_n=alpha_n,
    )

    # 密度剖面（同样用 ψ_N，略宽一点：σ_n 效果用 alpha_m 调小）
    psiN_grid = np.array(grid_data["psiN_profile"])
    n_profile  = np.zeros_like(psiN_grid)
    core_mask  = psiN_grid <= 1.0
    psiN_c     = np.clip(psiN_grid[core_mask], 0.0, 1.0)
    n_profile[core_mask] = n_e * (1.0 - psiN_c ** 0.7) ** 1.5   # 宽一点的密度剖面
    sol_mask   = psiN_grid > 1.0
    n_profile[sol_mask] = n_e * 0.01 * np.exp(-(psiN_grid[sol_mask] - 1.0) / 0.06)
    n_profile  = np.clip(n_profile, 0.0, None)

    return {
        "T_profile":    grid_data["T_profile"],
        "psiN_profile": grid_data["psiN_profile"],
        "n_profile":    n_profile.tolist(),
        "r_grid":       grid_data["r_grid"],
        "theta_grid":   grid_data["theta_grid"],
        "x_grid":       grid_data["x_grid"],
        "y_grid":       grid_data["y_grid"],
        "gs_meta":      meta,
    }


def generate_training_dataset_gs(
    n_samples:  int   = 200,
    grid_size:  int   = 8,
    save_path:  str   = None,
) -> list:
    """
    用 GS 平衡生成训练数据集（比纯解析高斯数据物理上更真实）

    注意：GS 求解每次约 1-3s，n_samples=200 约需 5-10 分钟
    建议先用小 n_samples 验证，再扩大规模

    每条样本：(n_e, T_e, B, r, theta) → T_at_point（基于 ψ_N 剖面）
    """
    import json, os

    np.random.seed(42)
    n_e_samples = np.power(10, np.random.uniform(18, 20, n_samples))
    T_e_samples = np.power(10, np.random.uniform(6,  8,  n_samples))
    B_samples   = np.random.uniform(1, 10, n_samples)

    r_values     = np.linspace(0, 1, grid_size)
    theta_values = np.linspace(0, 2 * np.pi, grid_size)

    dataset = []
    print(f"[GS] 开始生成 GS 平衡数据集 {n_samples} 组 × {grid_size}² 点（每组约 2s）...")

    for i in range(n_samples):
        n_e = n_e_samples[i]
        T_e = T_e_samples[i]
        B   = B_samples[i]

        try:
            eq, meta = solve_gs_equilibrium(n_e, T_e, B, nx=17, ny=17)
            gd = build_temperature_on_grid(eq, T_e=T_e, grid_size=grid_size)
            T_2d    = np.array(gd["T_profile"])
            psiN_2d = np.array(gd["psiN_profile"])

            for ti, theta in enumerate(theta_values):
                for ri, r in enumerate(r_values):
                    dataset.append({
                        "n_e":    float(n_e),
                        "T_e":    float(T_e),
                        "B":      float(B),
                        "r":      float(r),
                        "theta":  float(theta),
                        "T_out":  float(T_2d[ti, ri]),
                        "n_out":  0.0,
                        "psiN":   float(psiN_2d[ti, ri]),
                        "beta":   float(meta["poloidal_beta"]),
                        "lambda_D": 0.0,
                    })

        except Exception as e:
            print(f"  [警告] 第 {i} 组参数求解失败，跳过: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  已完成 {i+1}/{n_samples} 组")

    print(f"[GS] 数据集生成完成，共 {len(dataset)} 条样本")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(dataset, f)
        print(f"[GS] 已保存到 {save_path}")

    return dataset


# ─── 快速验证入口 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== FreeGS G-S 平衡验证 ===")
    n_e, T_e, B = 1e19, 1e7, 5.0

    eq, meta = solve_gs_equilibrium(n_e, T_e, B)
    print(f"  轴压      p_axis = {meta['p_axis']:.3e} Pa")
    print(f"  等离子体电流 I_p = {meta['plasma_current']:.3e} A")
    print(f"  极向 β          = {meta['poloidal_beta']:.4f}")
    print(f"  psi_axis        = {meta['psi_axis']:.4f}")
    print(f"  psi_bndry       = {meta['psi_bndry']:.4f}")
    if meta["q95"]:
        print(f"  q95             = {meta['q95']:.2f}")

    gd = build_temperature_on_grid(eq, T_e=T_e, grid_size=16)
    T  = np.array(gd["T_profile"])
    pN = np.array(gd["psiN_profile"])
    print(f"\n  温度剖面: T_max={T.max():.3e} K, T_min={T.min():.3e} K")
    print(f"  ψ_N 范围: [{pN.min():.3f}, {pN.max():.3f}]")
    print("=== 验证完成 ===")
