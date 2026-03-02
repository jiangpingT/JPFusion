"""
world_model.py — MLP Ensemble 世界模型（Phase 4）

5 个独立 MLP，输入 (state[7], action[3]) → 输出 (next_state[7], reward[1])
均值 = 最优预测，方差 = 不确定性估计

不确定性高的区域 → 优先从真实数据学习（好奇心驱动）
不确定性低的区域 → 用虚拟轨迹训练 RL Agent（Dyna 循环）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent.parent / "data"
WM_DIR   = DATA_DIR / "rl_models" / "world_model"
WM_DIR.mkdir(parents=True, exist_ok=True)

# 全局世界模型实例（供 mbrl_train 和 API 使用）
_world_model: Optional["WorldModelEnsemble"] = None


def get_world_model() -> Optional["WorldModelEnsemble"]:
    return _world_model


class SingleMLP(nn.Module):
    """单个 MLP 预测模型（状态 + 动作 → 下一步状态 + 奖励）"""

    def __init__(self, in_dim: int = 10, out_dim: int = 8, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WorldModelEnsemble(nn.Module):
    """
    MLP Ensemble 世界模型

    n_models=5 个独立 MLP，各自预测下一步。
    ensemble 均值 = 主预测；ensemble 方差 = 不确定性。
    """

    def __init__(self, state_dim: int = 7, action_dim: int = 3,
                 n_models: int = 5, hidden: int = 256):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.n_models   = n_models
        self.out_dim    = state_dim + 1  # next_state + reward

        self.models = nn.ModuleList([
            SingleMLP(state_dim + action_dim, self.out_dim, hidden)
            for _ in range(n_models)
        ])

    def forward(self, state: torch.Tensor, action: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: (B, out_dim) — 均值预测（next_state + reward）
            var:  (B, out_dim) — 方差（不确定性）
        """
        x = torch.cat([state, action], dim=-1)
        preds = torch.stack([m(x) for m in self.models], dim=0)  # (n_models, B, out_dim)
        mean  = preds.mean(dim=0)
        var   = preds.var(dim=0)
        return mean, var

    def predict_next(self, state: np.ndarray, action: np.ndarray,
                     device: str = None) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        单步预测（numpy 接口，供 WorldModelEnv 使用）

        Returns:
            next_state_mean: (7,)
            next_state_std:  (7,)
            reward_mean:     float
            uncertainty:     float（全局不确定性标量，越高越不确定）
        """
        # 自动检测模型所在设备
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            s = torch.tensor(state,  dtype=torch.float32).unsqueeze(0).to(device)
            a = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
            mean, var = self(s, a)

        mean_np = mean[0].cpu().numpy()
        var_np  = var[0].cpu().numpy()

        next_state_mean = np.clip(mean_np[:self.state_dim], 0.0, 1.0)
        next_state_std  = np.sqrt(np.clip(var_np[:self.state_dim], 0.0, 1.0))
        reward_mean     = float(mean_np[self.state_dim])
        uncertainty     = float(var_np.mean())

        return next_state_mean, next_state_std, reward_mean, uncertainty


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_world_model(
    east_data_path: str = None,
    east_df=None,
    n_epochs: int    = 100,
    n_models: int    = 5,
    lr: float        = 3e-4,
    batch_size: int  = 256,
    val_ratio: float = 0.1,
) -> "WorldModelEnsemble":
    """
    从 EAST 数据训练世界模型。

    可接受文件路径或已加载的 DataFrame。
    Returns: 训练好的 WorldModelEnsemble
    """
    global _world_model

    from backend.data.east_loader import load_itpa_iddb, load_synthetic_east_data
    from backend.data.replay_buffer import build_replay_buffer

    if east_df is None:
        print(f"[WorldModel] 加载数据：{east_data_path}")
        try:
            east_df = load_itpa_iddb(east_data_path)
        except Exception as e:
            print(f"[WorldModel] 真实数据加载失败（{e}），使用合成数据")
            east_df = load_synthetic_east_data(n_shots=100)

    buffer = build_replay_buffer(east_df)
    obs        = torch.tensor(buffer["observations"],      dtype=torch.float32)
    actions    = torch.tensor(buffer["actions"],           dtype=torch.float32)
    rewards    = torch.tensor(buffer["rewards"],           dtype=torch.float32).unsqueeze(1)
    next_obs   = torch.tensor(buffer["next_observations"], dtype=torch.float32)

    # 目标：next_state (7D) + reward (1D) = 8D
    targets = torch.cat([next_obs, rewards], dim=1)
    inputs  = torch.cat([obs, actions], dim=1)

    # Train / Val split
    n      = len(obs)
    n_val  = max(1, int(n * val_ratio))
    n_train = n - n_val
    perm   = torch.randperm(n)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    train_ds = TensorDataset(inputs[train_idx], targets[train_idx])
    val_ds   = TensorDataset(inputs[val_idx],   targets[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = _get_device()
    model  = WorldModelEnsemble(n_models=n_models).to(device)
    optimizers = [optim.Adam(m.parameters(), lr=lr) for m in model.models]
    criterion  = nn.MSELoss()

    print(f"\n[WorldModel] 开始训练 Ensemble（{n_models} 模型）")
    print(f"  训练：{n_train} | 验证：{n_val} | 设备：{device}")
    print(f"  轮次：{n_epochs} | batch：{batch_size} | lr：{lr}\n")

    best_val_loss = float("inf")
    save_path = str(WM_DIR / "world_model.pt")

    for epoch in range(1, n_epochs + 1):
        # ─── 训练（每个 MLP 独立更新）────────────────────────────────────
        for m_idx, (m, opt) in enumerate(zip(model.models, optimizers)):
            m.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = m(x_batch)
                loss = criterion(pred, y_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # ─── 验证（用 ensemble 均值）────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                state_b  = x_batch[:, :7]
                action_b = x_batch[:, 7:]
                mean, _  = model(state_b, action_b)
                val_losses.append(criterion(mean, y_batch).item())

        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 20 == 0 or epoch == n_epochs:
            print(f"  Epoch {epoch:>4d}/{n_epochs} | val_loss={val_loss:.5f} "
                  f"| best={best_val_loss:.5f}")

    # 加载最优权重
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    _world_model = model

    print(f"\n[WorldModel] 训练完成！最优 val_loss={best_val_loss:.5f}")
    print(f"[WorldModel] 模型已保存：{save_path}")
    pred_err_pct = best_val_loss ** 0.5 * 100  # 近似
    print(f"[WorldModel] 预测误差近似：{pred_err_pct:.1f}%（目标 < 5%）")

    return model


def get_uncertainty_map(n_grid: int = 20) -> dict:
    """
    在 (n_e_norm, P_heat_norm) 二维网格上计算世界模型不确定性热图。

    Returns:
        {
            "n_e_axis":    [float],   # X 轴
            "P_heat_axis": [float],   # Y 轴
            "uncertainty": [[float]], # 不确定性矩阵
        }
    """
    global _world_model

    if _world_model is None:
        save_path = str(WM_DIR / "world_model.pt")
        if not Path(save_path).exists():
            raise RuntimeError("世界模型尚未训练，请先调用 POST /api/rl/train-world-model")
        device = _get_device()
        _world_model = WorldModelEnsemble().to(device)
        _world_model.load_state_dict(torch.load(save_path, map_location=device))
        _world_model.eval()

    device = _get_device()
    n_vals = np.linspace(0.0, 1.0, n_grid)
    p_vals = np.linspace(0.0, 1.0, n_grid)
    uncertainty = np.zeros((n_grid, n_grid))

    # 固定其他维度（中间值）
    fixed_state  = np.array([0.3, 0.2, 0.5, 0.5, 0.3, 0.4, 0.3], dtype=np.float32)
    fixed_action = np.zeros(3, dtype=np.float32)

    for i, n_norm in enumerate(n_vals):
        for j, p_norm in enumerate(p_vals):
            state = fixed_state.copy()
            state[0] = n_norm  # n_e_norm
            state[6] = p_norm  # P_heat_norm
            _, _, _, unc = _world_model.predict_next(state, fixed_action)
            uncertainty[i, j] = unc

    return {
        "n_e_axis":    n_vals.tolist(),
        "P_heat_axis": p_vals.tolist(),
        "uncertainty": uncertainty.tolist(),
        "min_uncertainty": float(uncertainty.min()),
        "max_uncertainty": float(uncertainty.max()),
    }
