"""
physics_calibrator.py — τ_E 系数拟合（Phase 2A）

用 EAST 真实放电数据，拟合 ITER98pY2 能量约束时间经验公式系数：
    τ_E = C × n^a × B^c × P^d × Ip^e

让 FusionEnv 的动力学更接近真实 EAST 行为。
验证标准：R² > 0.8（物理上有意义的拟合）
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Optional
from backend.rl.dynamics import TAU_E_COEFFICIENTS


def _tau_e_model(X, C, a, c, d, e):
    """
    ITER98pY2 简化公式（对数线性化后的拟合函数）

    X = (n_e_19, B, P_MW, Ip_MA)
    """
    n_e_19, B, P_MW, Ip_MA = X
    # 使用 np.clip 避免 log(0) 或负数问题
    n_e_19 = np.clip(n_e_19, 1e-3, 1e3)
    B      = np.clip(B, 1e-3, 20.0)
    P_MW   = np.clip(P_MW, 1e-3, 1e4)
    Ip_MA  = np.clip(Ip_MA, 1e-3, 100.0)

    R      = TAU_E_COEFFICIENTS["R"]
    a_min  = TAU_E_COEFFICIENTS["a_minor"]
    kappa  = TAU_E_COEFFICIENTS["kappa"]
    M      = TAU_E_COEFFICIENTS["M"]

    return (C
            * (n_e_19 ** a)
            * (B ** c)
            * (P_MW ** d)
            * (Ip_MA ** e)
            * (R ** 1.97)
            * (a_min ** 0.58)
            * (kappa ** 0.78)
            * (M ** 0.19))


def fit_tau_e_coefficients(df: pd.DataFrame) -> dict:
    """
    从 EAST 放电数据拟合 τ_E 系数。

    输入 DataFrame 需包含列：n_e, B, P_heat, Ip, tau_E（可以有 NaN）
    Returns:
        {
            "coefficients": {...},  # 拟合系数
            "r_squared": float,     # 拟合优度 R²
            "n_samples": int,       # 有效数据点数
            "success": bool,        # R² > 0.8 则认为成功
            "message": str,
        }
    """
    required = ["n_e", "B", "P_heat", "Ip", "tau_E"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"DataFrame 缺少列: {col}（需要: {required}）")

    # ─── 过滤有效数据 ─────────────────────────────────────────────────────
    mask = (
        df["n_e"].notna() & (df["n_e"] > 0) &
        df["B"].notna()   & (df["B"] > 0) &
        df["P_heat"].notna() & (df["P_heat"] > 0) &
        df["Ip"].notna()  & (df["Ip"] > 0) &
        df["tau_E"].notna() & (df["tau_E"] > 0) & (df["tau_E"] < 10.0)
    )
    valid = df[mask].copy()

    if len(valid) < 20:
        return {
            "coefficients": TAU_E_COEFFICIENTS,
            "r_squared":    0.0,
            "n_samples":    len(valid),
            "success":      False,
            "message":      f"有效数据只有 {len(valid)} 条（需要 ≥ 20），使用默认系数",
        }

    # ─── 准备拟合数据 ─────────────────────────────────────────────────────
    n_e_19 = valid["n_e"].values / 1e19
    B      = valid["B"].values
    P_MW   = valid["P_heat"].values / 1e6
    Ip_MA  = valid["Ip"].values / 1e6
    tau_E  = valid["tau_E"].values

    X = (n_e_19, B, P_MW, Ip_MA)

    # ─── 初始猜测 = ITER98pY2 默认系数 ────────────────────────────────────
    p0 = [
        TAU_E_COEFFICIENTS["C"],  # C
        TAU_E_COEFFICIENTS["a"],  # n 指数
        TAU_E_COEFFICIENTS["c"],  # B 指数
        TAU_E_COEFFICIENTS["d"],  # P 指数
        TAU_E_COEFFICIENTS["e"],  # Ip 指数
    ]

    bounds = (
        [1e-4, 0.0, 0.0, -2.0, 0.0],    # 下界
        [1.0,  1.0, 1.0,  0.0, 2.0],    # 上界
    )

    try:
        popt, pcov = curve_fit(
            _tau_e_model, X, tau_E,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
        C_fit, a_fit, c_fit, d_fit, e_fit = popt

        # R² 计算
        tau_pred = _tau_e_model(X, *popt)
        ss_res   = np.sum((tau_E - tau_pred) ** 2)
        ss_tot   = np.sum((tau_E - np.mean(tau_E)) ** 2)
        r2       = float(1.0 - ss_res / max(ss_tot, 1e-30))

        fitted_coeffs = {
            **TAU_E_COEFFICIENTS,
            "C": float(C_fit),
            "a": float(a_fit),
            "c": float(c_fit),
            "d": float(d_fit),
            "e": float(e_fit),
        }

        success = r2 > 0.8
        msg = (
            f"拟合成功（R²={r2:.4f}）" if success
            else f"拟合完成但 R²={r2:.4f} < 0.8，物理意义有限"
        )
        print(f"[physics_calibrator] {msg}")
        print(f"  C={C_fit:.4f}, n_exp={a_fit:.3f}, B_exp={c_fit:.3f}, "
              f"P_exp={d_fit:.3f}, Ip_exp={e_fit:.3f}")

        return {
            "coefficients": fitted_coeffs,
            "r_squared":    r2,
            "n_samples":    len(valid),
            "success":      success,
            "message":      msg,
        }

    except Exception as e:
        return {
            "coefficients": TAU_E_COEFFICIENTS,
            "r_squared":    0.0,
            "n_samples":    len(valid),
            "success":      False,
            "message":      f"拟合失败: {e}，使用默认系数",
        }
