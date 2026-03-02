"""
bc_pretrain.py — Phase 2 · Sim-SFT：仿真环境监督微调（行为克隆 BC）

【阶段定位】
  数据来源：专家轨迹（EAST 放电数据 / Phase 1 高质量轨迹）
  学习方式：SFT（监督微调）= BC（行为克隆），MSE 拟合专家动作
  在四阶段流水线中的角色：
    → 输出 SFT Actor 权重（p2_sft_actor.pt），热启动 Phase 1 · Sim-RL

【类比】类似 LLM 的 SFT：先让模型学会"说人话"，再用 RL 精炼
  SFT 冷启动 → Phase 1 PPO 热启动，收敛速度约快 2×

专家策略来源：
  状态输入（7D）：n_e, T_e, B, q95, beta_N, Ip, P_heat
  动作输出（3D）：ΔP_heat, Δn_e, ΔIp

运行命令：
    python -m backend.rl.bc_pretrain --east_data data/east_discharge.csv
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from backend.rl.dynamics import PHYSICS_RANGES, denormalize_state

DATA_DIR     = Path(__file__).parent.parent.parent / "data"
BC_MODEL_DIR = DATA_DIR / "rl_models" / "p2_sft"
BC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# BC/SFT 训练状态（供外部查询）
_bc_state = {
    "status":     "idle",
    "epoch":      0,
    "n_epochs":   0,
    "train_loss": None,
    "val_loss":   None,
    "model_path": None,
}


def get_bc_state() -> dict:
    return dict(_bc_state)


class EASTExpertDataset(Dataset):
    """
    EAST 专家 (state, action) 对 数据集

    从 east_loader.py 输出的 DataFrame 提取：
      - state: 7D 归一化向量（n_e_norm, T_e_norm, B_norm, q95_norm, beta_N_norm, Ip_norm, P_heat_norm）
      - action: 3D 控制量增量（由相邻时间步差分得到）
    """

    def __init__(self, df: pd.DataFrame):
        self.states  = []
        self.actions = []

        n_lo, n_hi   = PHYSICS_RANGES["n_e"]
        T_lo, T_hi   = PHYSICS_RANGES["T_e"]
        B_lo,  B_hi  = PHYSICS_RANGES["B"]
        q_lo,  q_hi  = PHYSICS_RANGES["q95"]
        bN_lo, bN_hi = PHYSICS_RANGES["beta_N"]
        Ip_lo, Ip_hi = PHYSICS_RANGES["Ip"]
        P_lo,  P_hi  = PHYSICS_RANGES["P_heat"]

        for shot_id in df["shot_id"].unique():
            shot = df[df["shot_id"] == shot_id].sort_values("time").reset_index(drop=True)
            if len(shot) < 2:
                continue

            for i in range(len(shot) - 1):
                row  = shot.iloc[i]
                next_row = shot.iloc[i + 1]

                # ─── 状态归一化 ──────────────────────────────────────────
                def norm(val, lo, hi):
                    return float(np.clip((val - lo) / max(hi - lo, 1e-30), 0.0, 1.0))

                # q95 和 beta_N 从数据中读取，如缺失则用默认
                from backend.rl.dynamics import compute_q95, compute_beta_N
                Ip  = float(row.get("Ip", 1e6))
                B   = float(row.get("B", 4.0))
                n_e = float(row.get("n_e", 3e19))
                T_e = float(row.get("T_e", 1e7))
                q95   = float(row["q95"])   if "q95" in row and not np.isnan(row["q95"])  else compute_q95(Ip, B)
                beta_N = float(row["beta_N"]) if "beta_N" in row and "beta_N" in df.columns and not np.isnan(row.get("beta_N", float("nan"))) else compute_beta_N(n_e, T_e, B, Ip)
                P_heat = float(row.get("P_heat", 5e6))

                state = np.array([
                    norm(n_e,    n_lo,  n_hi),
                    norm(T_e,    T_lo,  T_hi),
                    norm(B,      B_lo,  B_hi),
                    norm(q95,    q_lo,  q_hi),
                    norm(beta_N, bN_lo, bN_hi),
                    norm(Ip,     Ip_lo, Ip_hi),
                    norm(P_heat, P_lo,  P_hi),
                ], dtype=np.float32)

                # ─── 动作：读取预存或从差分计算 ──────────────────────────
                if "action_P_heat" in df.columns:
                    action = np.array([
                        float(row.get("action_P_heat", 0)),
                        float(row.get("action_n_fuel", 0)),
                        float(row.get("action_Ip", 0)),
                    ], dtype=np.float32)
                else:
                    # 差分方式：Δ(归一化值)
                    P_next  = float(next_row.get("P_heat", P_heat))
                    n_next  = float(next_row.get("n_e",    n_e))
                    Ip_next = float(next_row.get("Ip",     Ip))
                    action = np.array([
                        np.clip((P_next  - P_heat) / max(P_hi  - P_lo,  1.0), -0.1, 0.1),
                        np.clip((n_next  - n_e)    / max(n_hi  - n_lo,  1.0), -0.1, 0.1),
                        np.clip((Ip_next - Ip)     / max(Ip_hi - Ip_lo, 1.0), -0.1, 0.1),
                    ], dtype=np.float32)

                action = np.clip(action, -0.1, 0.1)
                self.states.append(state)
                self.actions.append(action)

        self.states  = np.array(self.states,  dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        print(f"[BC/SFT] 数据集大小：{len(self.states)} 个 (state, action) 对")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.states[idx]),
            torch.tensor(self.actions[idx]),
        )


class BCActorMLP(nn.Module):
    """
    SFT 行为克隆 Actor 网络（MLP）

    输入：7D 状态向量
    输出：3D 动作向量（[-0.1, 0.1] 范围，用 tanh × 0.1 激活）
    """

    def __init__(self, state_dim: int = 7, action_dim: int = 3,
                 hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )
        self._action_scale = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self._action_scale


def behavior_clone_sft(
    east_data_path: str = None,
    east_df: pd.DataFrame = None,
    n_epochs: int   = 500,
    lr: float       = 1e-3,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    save_path: str  = None,
) -> str:
    """
    SFT 行为克隆预训练主函数。

    可接受文件路径（east_data_path）或已加载的 DataFrame（east_df）。
    Returns: 保存的权重路径
    """
    _bc_state["status"]   = "training"
    _bc_state["n_epochs"] = n_epochs

    if east_df is None:
        from backend.data.east_loader import load_itpa_iddb
        east_df = load_itpa_iddb(east_data_path)

    # ─── 构建数据集 ───────────────────────────────────────────────────────
    dataset = EASTExpertDataset(east_df)
    n_val   = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    # ─── 模型 + 优化器 ─────────────────────────────────────────────────────
    device = torch.device("cpu")  # M4 Pro 无 CUDA，用 MPS 或 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    model     = BCActorMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    if save_path is None:
        save_path = str(BC_MODEL_DIR / "p2_sft_actor.pt")

    print(f"\n[BC/SFT] 开始 SFT 行为克隆预训练")
    print(f"  训练集：{n_train} 样本 | 验证集：{n_val} 样本")
    print(f"  设备：{device} | 轮次：{n_epochs} | 学习率：{lr}")
    print(f"  保存路径：{save_path}\n")

    for epoch in range(1, n_epochs + 1):
        # ─── 训练 ────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            pred = model(states)
            loss = criterion(pred, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ─── 验证 ────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                pred = model(states)
                val_losses.append(criterion(pred, actions).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        scheduler.step()

        _bc_state["epoch"]      = epoch
        _bc_state["train_loss"] = train_loss
        _bc_state["val_loss"]   = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 50 == 0 or epoch == n_epochs:
            print(f"  Epoch {epoch:>4d}/{n_epochs} | "
                  f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

    _bc_state["status"]     = "done"
    _bc_state["model_path"] = save_path

    print(f"\n[BC/SFT] 训练完成！最优 val_loss={best_val_loss:.5f}")
    print(f"[BC/SFT] 权重已保存到：{save_path}")
    print("\n【名词备注】")
    print("  SFT（监督微调 Supervised Fine-Tuning）= 行为克隆（BC）")
    print("  — what: 用专家轨迹做监督学习，让模型先学会基础控制策略")
    print("  — why:  避免 PPO 从随机策略开始漫无目的探索，节省训练时间")
    print("  — how:  SFT 后 val_loss < 1e-4 通常效果较好，再热启动 PPO")

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT 行为克隆预训练")
    parser.add_argument("--east_data", type=str, required=True)
    parser.add_argument("--n_epochs",  type=int, default=500)
    parser.add_argument("--lr",        type=float, default=1e-3)
    args = parser.parse_args()

    behavior_clone_sft(
        east_data_path=args.east_data,
        n_epochs=args.n_epochs,
        lr=args.lr,
    )
