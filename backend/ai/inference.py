"""
inference.py — 模型推理接口

职责：
  1. 加载训练好的模型
  2. 给定 (n_e, T_e, B)，在 grid_size×grid_size 的截面网格上逐点推理
  3. 支持 MC Dropout 不确定性估计（多次前向传播取均值和方差）
  4. 返回可直接传给前端 Plotly 的二维数组数据
"""

import os
import numpy as np
import torch
from typing import Dict, Optional

from backend.ai.model import load_model, PlasmaFieldMLP
from backend.data.dataset import normalize_input, denormalize_output_T

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "../../data/plasma_model.pt")

# ─── 缓存已加载的模型（避免每次推理重新加载）───────────────────────────────
_cached_model: Optional[PlasmaFieldMLP] = None
_cached_device: str = "cpu"


def get_model() -> Optional[PlasmaFieldMLP]:
    """获取已缓存的模型（如果模型文件存在）"""
    global _cached_model, _cached_device
    model_path = os.path.abspath(MODEL_SAVE_PATH)
    if not os.path.exists(model_path):
        return None
    if _cached_model is None:
        _cached_device = "mps" if torch.backends.mps.is_available() else "cpu"
        _cached_model  = load_model(model_path, device=_cached_device)
    return _cached_model


def reload_model():
    """训练完成后重新加载模型"""
    global _cached_model
    _cached_model = None
    return get_model()


def run_inference(
    n_e: float,
    T_e: float,
    B:   float,
    grid_size: int = 32,
    use_physics_fallback: bool = True,
    use_mc_dropout: bool = True,
    n_mc_samples:   int  = 20,
) -> Dict:
    """
    在 grid_size×grid_size 的截面上运行推理

    参数：
      n_e, T_e, B          — 等离子体参数（原始物理量级）
      grid_size            — 网格分辨率
      use_physics_fallback — 模型未训练时，用物理解析解代替
      use_mc_dropout       — 启用 MC Dropout 不确定性估计
      n_mc_samples         — MC Dropout 采样次数（默认 20 次）

    返回：
      {
        "source":        "model" | "physics",
        "x":             [...],          # 网格 x 坐标
        "y":             [...],          # 网格 y 坐标
        "T_values":      [...],          # 温度均值（grid_size × grid_size）
        "T_uncertainty": [...] | None,   # 温度标准差（MC Dropout，仅 AI 模式）
        "T_min":         float,
        "T_max":         float,
        "grid_size":     int,
        "physics_params": {...},
      }
    """
    model = get_model()

    # 极坐标网格
    r_lin     = np.linspace(0, 1, grid_size)
    theta_lin = np.linspace(0, 2 * np.pi, grid_size)
    r_grid, theta_grid = np.meshgrid(r_lin, theta_lin)

    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    T_uncertainty_grid = None
    psiN_result        = None
    gs_meta            = None

    if model is not None:
        # ── 模型推理路径 ───────────────────────────────────────────────────
        r_flat     = r_grid.flatten()
        theta_flat = theta_grid.flatten()
        n_pts      = len(r_flat)

        # 批量构造输入（一次性，避免重复）
        inputs = np.stack([
            normalize_input(n_e, T_e, B, r_flat[i], theta_flat[i])
            for i in range(n_pts)
        ])  # shape (n_pts, 5)
        X = torch.tensor(inputs, dtype=torch.float32).to(_cached_device)

        if use_mc_dropout and n_mc_samples > 1:
            # ── MC Dropout：保持 train() 模式让 Dropout 激活，多次采样 ─────
            model.train()   # Dropout 在 eval() 时关闭，train() 时激活
            all_preds = []
            with torch.no_grad():
                for _ in range(n_mc_samples):
                    preds_norm = model(X).cpu().numpy().flatten()
                    preds_K    = np.array([denormalize_output_T(v) for v in preds_norm])
                    all_preds.append(preds_K)

            all_preds  = np.stack(all_preds, axis=0)    # (n_mc, n_pts)
            mean_preds = all_preds.mean(axis=0)          # (n_pts,)
            std_preds  = all_preds.std(axis=0)           # (n_pts,) 不确定性

            T_grid             = mean_preds.reshape(grid_size, grid_size)
            T_uncertainty_grid = std_preds.reshape(grid_size, grid_size)
            model.eval()    # 恢复 eval 模式
        else:
            # 标准推理（单次）
            model.eval()
            with torch.no_grad():
                preds_norm = model(X).cpu().numpy().flatten()
            preds_K = np.array([denormalize_output_T(v) for v in preds_norm])
            T_grid  = preds_K.reshape(grid_size, grid_size)

        source = "model"

    elif use_physics_fallback:
        # ── 物理解析解：优先用 GS 平衡，失败时退回高斯 ─────────────────────
        psiN_result = None
        gs_meta     = None
        try:
            from backend.physics.gs_engine import solve_gs_equilibrium, build_temperature_on_grid
            eq, gs_meta = solve_gs_equilibrium(n_e, T_e, B, nx=33, ny=33)
            gd       = build_temperature_on_grid(eq, T_e=T_e, grid_size=grid_size)
            T_grid   = np.array(gd["T_profile"])
            psiN_result = np.array(gd["psiN_profile"])
            source   = "physics_gs"
        except Exception as e:
            print(f"[Inference] GS 求解失败（退回高斯）: {e}")
            sigma_T  = 0.30
            r_sep    = 0.90
            lambda_q = 0.03
            T_grid   = T_e * np.exp(-r_grid**2 / (2 * sigma_T**2))
            T_lcfs   = T_e * np.exp(-r_sep**2 / (2 * sigma_T**2))
            sol_mask = r_grid > r_sep
            T_grid[sol_mask] = T_lcfs * np.exp(-(r_grid[sol_mask] - r_sep) / lambda_q)
            source   = "physics"
    else:
        raise RuntimeError("模型未训练，且不允许物理后备模式")

    # ── 计算标量物理量 ─────────────────────────────────────────────────────
    from backend.physics.plasma_engine import compute_scalar_physics
    physics = compute_scalar_physics(n_e, T_e, B)

    result = {
        "source":     source,
        "x":          x_grid.tolist(),
        "y":          y_grid.tolist(),
        "T_values":   T_grid.tolist(),
        "T_min":      float(T_grid.min()),
        "T_max":      float(T_grid.max()),
        "grid_size":  grid_size,
        "physics_params": {
            "lambda_D":  float(physics["lambda_D"]),
            "omega_p":   float(physics["omega_p"]),
            "v_alfven":  float(physics["v_alfven"]),
            "beta":      float(physics["beta"]),
        },
    }

    # GS 平衡额外数据（仅 physics_gs 来源时存在）
    if source == "physics_gs" and psiN_result is not None:
        result["psiN_profile"] = psiN_result.tolist()
        result["gs_meta"]      = gs_meta
    else:
        result["psiN_profile"] = None
        result["gs_meta"]      = None

    if T_uncertainty_grid is not None:
        result["T_uncertainty"]      = T_uncertainty_grid.tolist()
        result["T_uncertainty_max"]  = float(T_uncertainty_grid.max())
        result["T_uncertainty_mean"] = float(T_uncertainty_grid.mean())
    else:
        result["T_uncertainty"]      = None
        result["T_uncertainty_max"]  = None
        result["T_uncertainty_mean"] = None

    return result
