"""
trainer.py — 训练循环 + WebSocket 实时进度推送

流程：
  1. 加载数据集
  2. 创建 / 加载模型
  3. 每个 Epoch 后通过 WebSocket 推送 JSON 指标
  4. 支持异步运行（在 asyncio 事件循环中执行训练）
  5. 支持 PINN（Physics-Informed Neural Network）物理约束损失
  6. 训练状态持久化到 SQLite（服务重启后可恢复）

WebSocket 推送格式：
{
  "type": "training_progress",
  "epoch": 15,
  "total_epochs": 100,
  "train_loss": 0.0023,
  "val_loss": 0.0031,
  "train_mae": 0.04,
  "val_mae": 0.05,
  "pde_loss": 0.0005,
  "lr": 0.001,
  "elapsed_sec": 12.5,
  "status": "training"  // "training" | "completed" | "error"
}
"""

import asyncio
import time
import os
import json
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Callable, Awaitable, Optional

from backend.ai.model import create_model, save_model, PlasmaFieldMLP
from backend.data.dataset import load_dataset

# ─── 全局训练状态（单例，供 API 查询）─────────────────────────────────────
_training_state = {
    "status":       "idle",   # idle | generating | training | completed | error
    "epoch":        0,
    "total_epochs": 0,
    "train_loss":   None,
    "val_loss":     None,
    "best_val_loss": float("inf"),
    "error_msg":    None,
    "history":      [],       # 每个 epoch 的完整日志
}

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "../../data/plasma_model.pt")
DATA_PATH       = os.path.join(os.path.dirname(__file__), "../../data/dataset.json")
DB_PATH         = os.path.join(os.path.dirname(__file__), "../../data/training.db")


# ─── SQLite 持久化 ─────────────────────────────────────────────────────────

def _save_state_to_db(state: dict):
    """将训练状态序列化保存到 SQLite（下次启动时可恢复）"""
    db_path = os.path.abspath(DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    # 只保存可序列化的字段（history 可能很大，截取最近 200 条）
    state_to_save = {k: v for k, v in state.items() if k != "history"}
    state_to_save["history"] = state.get("history", [])[-200:]
    # float("inf") 不可 JSON 序列化，转为 None
    if state_to_save.get("best_val_loss") == float("inf"):
        state_to_save["best_val_loss"] = None
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state
                (key TEXT PRIMARY KEY, value TEXT)
            """)
            conn.execute(
                "INSERT OR REPLACE INTO state VALUES ('training_state', ?)",
                [json.dumps(state_to_save)]
            )
    except Exception as e:
        print(f"[Trainer] SQLite 保存失败: {e}")


def _load_state_from_db() -> dict:
    """从 SQLite 恢复训练状态，失败时返回空字典"""
    db_path = os.path.abspath(DB_PATH)
    if not os.path.exists(db_path):
        return {}
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT value FROM state WHERE key='training_state'"
            ).fetchone()
        if row:
            loaded = json.loads(row[0])
            # 恢复 float("inf")
            if loaded.get("best_val_loss") is None:
                loaded["best_val_loss"] = float("inf")
            return loaded
    except Exception as e:
        print(f"[Trainer] SQLite 加载失败: {e}")
    return {}


def restore_state_from_db():
    """启动时从 SQLite 恢复训练状态（由 main.py 在 startup 时调用）"""
    global _training_state
    loaded = _load_state_from_db()
    if loaded:
        # 只恢复非 training 状态（training 中断视为 error）
        if loaded.get("status") == "training":
            loaded["status"] = "error"
            loaded["error_msg"] = "服务重启，训练中断"
        _training_state.update(loaded)
        print(f"[Trainer] 从 SQLite 恢复状态: status={loaded.get('status')}, "
              f"epoch={loaded.get('epoch')}, history={len(loaded.get('history', []))} 条")
    return _training_state


def get_training_state() -> dict:
    return dict(_training_state)


# ─── PINN 物理约束损失 ─────────────────────────────────────────────────────

def compute_physics_loss(
    model: nn.Module,
    X_batch: torch.Tensor,
    device: str,
    lambda_pde:  float = 0.01,
    lambda_mono: float = 0.05,
    lambda_sym:  float = 0.02,
    lambda_bc:   float = 0.03,
) -> dict:
    """
    PINN 物理约束损失（Physics-Informed Neural Network）

    四项物理约束：
    ① PDE 残差：∂²T/∂r² + (1/r)·∂T/∂r = 0（柱坐标稳态热扩散方程）
    ② 单调性：  dT/dr < 0（温度从中心向外单调递减）
    ③ 方位角对称：dT/dθ ≈ 0（托卡马克近似轴对称）
    ④ 边界条件：∂T/∂r(r≈0) ≈ 0（中心对称）+ T(r≈1) ≈ 0（边界低温）

    参数：
      model      — 当前训练中的神经网络
      X_batch    — 当前批次输入 (batch, 5)，索引: [n_e_norm, T_e_norm, B_norm, r, theta]
      device     — 计算设备
      lambda_*   — 各约束项权重

    返回：dict，含 total/pde/mono/sym/bc 各分项损失（均为 torch.Tensor）
    """
    # 创建需要梯度的输入副本（不影响原始计算图）
    X_phys = X_batch.detach().clone().requires_grad_(True)
    T_pred = model(X_phys)  # shape: (batch, 1)

    ones = torch.ones_like(T_pred)

    # 一阶偏导数（对所有输入维度）
    grads_1 = autograd.grad(
        T_pred, X_phys,
        grad_outputs=ones,
        create_graph=True,   # 需要 create_graph=True 才能对一阶导数再求导
        retain_graph=True,
    )[0]  # shape: (batch, 5)

    dT_dr     = grads_1[:, 3:4]  # ∂T/∂r，r 在输入索引 3
    dT_dtheta = grads_1[:, 4:5]  # ∂T/∂θ，θ 在输入索引 4

    # 二阶偏导数（∂²T/∂r²，dT_dr 对 r 再求导）
    grads_2 = autograd.grad(
        dT_dr, X_phys,
        grad_outputs=torch.ones_like(dT_dr),
        create_graph=False,
        retain_graph=True,
    )[0]  # shape: (batch, 5)
    d2T_dr2 = grads_2[:, 3:4]

    r   = X_phys[:, 3:4]   # 归一化半径 [0,1]
    eps = 1e-4              # 避免 r=0 时除零

    # ① PDE 残差：柱坐标稳态热扩散 ∂²T/∂r² + (1/r)·∂T/∂r = 0
    pde_residual = d2T_dr2 + dT_dr / (r + eps)
    pde_loss = lambda_pde * (pde_residual ** 2).mean()

    # ② 单调性：惩罚 dT/dr > 0 的情况（ReLU 截取正值）
    mono_loss = lambda_mono * (torch.relu(dT_dr) ** 2).mean()

    # ③ 方位角对称性：dT/dθ 应接近 0
    sym_loss = lambda_sym * (dT_dtheta ** 2).mean()

    # ④ 边界条件
    r_sq = r.squeeze()
    center_mask   = r_sq < 0.05   # 中心区域：∂T/∂r ≈ 0
    boundary_mask = r_sq > 0.93   # 边界区域：T ≈ 0

    bc_center = (
        (dT_dr[center_mask] ** 2).mean()
        if center_mask.any()
        else torch.tensor(0.0, device=device)
    )
    bc_boundary = (
        (T_pred[boundary_mask] ** 2).mean()
        if boundary_mask.any()
        else torch.tensor(0.0, device=device)
    )
    bc_loss = lambda_bc * (0.5 * bc_center + 0.5 * bc_boundary)

    total_loss = pde_loss + mono_loss + sym_loss + bc_loss

    return {
        "total": total_loss,
        "pde":   pde_loss,
        "mono":  mono_loss,
        "sym":   sym_loss,
        "bc":    bc_loss,
    }


# ─── 训练入口 ──────────────────────────────────────────────────────────────

async def train_model(
    n_epochs:         int   = 100,
    batch_size:       int   = 512,
    lr:               float = 1e-3,
    ws_callback:      Optional[Callable[[dict], Awaitable[None]]] = None,
    max_data_samples: int   = 30000,
    use_pinn:         bool  = True,
    lambda_pde:       float = 0.01,
    lambda_mono:      float = 0.05,
    lambda_sym:       float = 0.02,
    lambda_bc:        float = 0.03,
):
    """
    异步训练入口

    参数：
      n_epochs         — 训练轮次
      batch_size       — 批次大小
      lr               — 初始学习率
      ws_callback      — 异步函数，每 epoch 后调用，传入指标字典
      max_data_samples — 最多加载多少条数据（内存限制）
      use_pinn         — 是否启用物理约束损失（PINN）
      lambda_pde       — PDE 残差权重
      lambda_mono      — 单调性约束权重
      lambda_sym       — 方位角对称权重
      lambda_bc        — 边界条件权重
    """
    global _training_state

    _training_state["status"]       = "training"
    _training_state["epoch"]        = 0
    _training_state["total_epochs"] = n_epochs
    _training_state["history"]      = []
    _training_state["error_msg"]    = None
    _training_state["best_val_loss"] = float("inf")
    _save_state_to_db(_training_state)

    try:
        # ─ 加载数据 ──────────────────────────────────────────────────────────
        data_path = os.path.abspath(DATA_PATH)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集不存在，请先调用 /api/generate-data")

        train_loader, val_loader, stats = load_dataset(
            data_path,
            val_ratio=0.1,
            batch_size=batch_size,
            max_samples=max_data_samples,
        )

        # ─ 创建模型 ───────────────────────────────────────────────────────────
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model  = create_model(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
        criterion = nn.MSELoss()

        pinn_str = f"PINN ON (λ_pde={lambda_pde}, λ_mono={lambda_mono})" if use_pinn else "PINN OFF"
        start_time = time.time()
        print(f"[Trainer] 开始训练 {n_epochs} epochs，设备: {device}，批次: {batch_size}，{pinn_str}")

        for epoch in range(1, n_epochs + 1):
            # ── 训练阶段 ─────────────────────────────────────────────────────
            model.train()
            train_losses, train_maes, phys_losses = [], [], []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()

                pred      = model(X_batch)
                data_loss = criterion(pred, y_batch)

                if use_pinn:
                    phys = compute_physics_loss(
                        model, X_batch, device,
                        lambda_pde=lambda_pde,
                        lambda_mono=lambda_mono,
                        lambda_sym=lambda_sym,
                        lambda_bc=lambda_bc,
                    )
                    total_loss = data_loss + phys["total"]
                    phys_losses.append(phys["total"].item())
                else:
                    total_loss = data_loss

                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(data_loss.item())
                train_maes.append(torch.mean(torch.abs(pred - y_batch)).item())

            # ── 验证阶段 ─────────────────────────────────────────────────────
            model.eval()
            val_losses, val_maes = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred    = model(X_batch)
                    loss    = criterion(pred, y_batch)
                    val_losses.append(loss.item())
                    val_maes.append(torch.mean(torch.abs(pred - y_batch)).item())

            scheduler.step()

            train_loss = float(np.mean(train_losses))
            val_loss   = float(np.mean(val_losses))
            train_mae  = float(np.mean(train_maes))
            val_mae    = float(np.mean(val_maes))
            pde_loss   = float(np.mean(phys_losses)) if phys_losses else 0.0
            current_lr = float(scheduler.get_last_lr()[0])
            elapsed    = time.time() - start_time

            # 保存最优模型
            if val_loss < _training_state["best_val_loss"]:
                _training_state["best_val_loss"] = val_loss
                save_path = os.path.abspath(MODEL_SAVE_PATH)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_model(model, save_path, meta={
                    "epoch":      epoch,
                    "val_loss":   val_loss,
                    "train_loss": train_loss,
                    "use_pinn":   use_pinn,
                })

            # ── 更新全局状态 ───────────────────────────────────────────────
            epoch_metrics = {
                "type":         "training_progress",
                "epoch":        epoch,
                "total_epochs": n_epochs,
                "train_loss":   round(train_loss, 6),
                "val_loss":     round(val_loss, 6),
                "train_mae":    round(train_mae, 6),
                "val_mae":      round(val_mae, 6),
                "pde_loss":     round(pde_loss, 6),
                "lr":           round(current_lr, 7),
                "elapsed_sec":  round(elapsed, 1),
                "status":       "training",
            }
            _training_state.update({
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
            })
            _training_state["history"].append(epoch_metrics)

            # 每 10 epoch 持久化一次状态
            if epoch % 10 == 0:
                _save_state_to_db(_training_state)

            # ── 推送 WebSocket ────────────────────────────────────────────
            if ws_callback:
                await ws_callback(epoch_metrics)

            # 每 10 epoch 打印一次日志
            if epoch % 10 == 0 or epoch == 1:
                pde_str = f" pde_loss={pde_loss:.5f}" if use_pinn else ""
                print(f"  Epoch {epoch:3d}/{n_epochs} | "
                      f"train_loss={train_loss:.5f} val_loss={val_loss:.5f}"
                      f"{pde_str} lr={current_lr:.2e} t={elapsed:.0f}s")

            # 让出事件循环（避免阻塞其他 async 操作）
            await asyncio.sleep(0)

        _training_state["status"] = "completed"
        _save_state_to_db(_training_state)

        final_msg = {
            "type":         "training_progress",
            "epoch":        n_epochs,
            "total_epochs": n_epochs,
            "train_loss":   _training_state["train_loss"],
            "val_loss":     _training_state["val_loss"],
            "status":       "completed",
        }
        if ws_callback:
            await ws_callback(final_msg)

        print(f"[Trainer] 训练完成！最优 val_loss={_training_state['best_val_loss']:.6f}")

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        _training_state["status"]    = "error"
        _training_state["error_msg"] = str(e)
        _save_state_to_db(_training_state)
        print(f"[Trainer] 训练出错: {err_msg}")
        if ws_callback:
            await ws_callback({
                "type":   "training_progress",
                "status": "error",
                "error":  str(e),
            })
        raise
