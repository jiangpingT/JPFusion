"""
world_model_deeponet.py — FusionDeepONet 世界模型（Phase 4 Path C）

DeepONet（深度算子网络）打破 MLP Ensemble 的 Markov 限制：
  Branch 网络：吃 s_0（初始状态）+ 完整动作历史（过去 K 步）
  Trunk  网络：吃时间坐标 t
  输出：next_state (7D) + reward (1D)

关键优势：
  Branch 始终以 s_0 为锚点，不累积递归误差
  历史感知：过去 K=20 步动作对预测有影响（打破 Markov 假设）

保存路径：data/rl_models/world_model_deeponet/
  fusion_deeponet_{i}.pt  — Ensemble 子模型
  deeponet_config.json    — 超参数配置
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple

# ─── 路径 ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DON_DIR  = DATA_DIR / "rl_models" / "world_model_deeponet"
DON_DIR.mkdir(parents=True, exist_ok=True)

# ─── 超参数 ────────────────────────────────────────────────────────────────────
STATE_DIM  = 7    # 等离子体状态维度
ACTION_DIM = 3    # 控制动作维度
K          = 20   # 动作历史长度（过去 K 步）
OUT_DIM    = 8    # 输出维度（7D next_state + 1D reward）
P          = 256  # Branch/Trunk 公共隐空间维度
MAX_STEPS  = 500  # episode 最大步数（时间归一化分母）

BRANCH_IN  = STATE_DIM + ACTION_DIM * K  # 7 + 3×20 = 67
TRUNK_IN   = 1                            # 归一化时间步 ∈ [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# 核心网络
# ─────────────────────────────────────────────────────────────────────────────

class FusionDeepONet(nn.Module):
    """
    DeepONet 核心网络（Branch + Trunk + einsum 输出）

    输入：
      branch_input: (batch, 67) — [s_0(7), a_0(3), ..., a_{K-1}(3)]
                    s_0 是 episode 起始真实状态（不随步数更新，无递归误差）
      trunk_input:  (batch, 1)  — [t / max_steps]，归一化时间步

    输出：
      (batch, 8)  — [next_state(7), reward(1)]

    组合方式：
      branch → (batch, OUT_DIM, P) = (batch, 8, 256)
      trunk  → (batch, P)          = (batch, 256)
      einsum('bop,bp->bo') → (batch, 8)
      + bias(8,) → (batch, 8)
    """

    def __init__(self, branch_in: int = BRANCH_IN, trunk_in: int = TRUNK_IN,
                 p: int = P, out_dim: int = OUT_DIM):
        super().__init__()
        self.p       = p
        self.out_dim = out_dim

        # Branch 网络：(67,) → (P × out_dim,) = (2048,)
        self.branch = nn.Sequential(
            nn.Linear(branch_in, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, p * out_dim),
        )

        # Trunk 网络：(1,) → (P,) = (256,)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, p),
        )

        # 可学习输出偏置（每个输出维度独立）
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, branch_input: torch.Tensor,
                trunk_input: torch.Tensor) -> torch.Tensor:
        """
        branch_input: (batch, 67)
        trunk_input:  (batch, 1)
        Returns:      (batch, 8)
        """
        batch = branch_input.shape[0]

        # Branch → reshape: (batch, out_dim, p) = (batch, 8, 256)
        b_out = self.branch(branch_input)                # (batch, p * out_dim)
        b_out = b_out.view(batch, self.out_dim, self.p)  # (batch, 8, 256)

        # Trunk → (batch, 256)
        t_out = self.trunk(trunk_input)  # (batch, 256)

        # einsum: (batch, 8, 256) × (batch, 256) → (batch, 8)
        out = torch.einsum('bop,bp->bo', b_out, t_out)  # (batch, 8)
        out = out + self.bias                             # + bias (8,)

        return out


class FusionDeepONetEnsemble(nn.Module):
    """
    3 个 FusionDeepONet 的集成，提供均值预测 + 不确定性估计。

    接口与 WorldModelEnsemble.predict_next() 对齐（但参数不同）。
    """

    def __init__(self, n_models: int = 3, **kwargs):
        super().__init__()
        self.n_models = n_models
        self.models   = nn.ModuleList([
            FusionDeepONet(**kwargs) for _ in range(n_models)
        ])

    def forward(self, branch_input: torch.Tensor,
                trunk_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: (batch, 8) — 均值预测（next_state + reward）
            var:  (batch, 8) — 方差（不确定性）
        """
        preds = torch.stack(
            [m(branch_input, trunk_input) for m in self.models], dim=0
        )  # (n_models, batch, 8)
        mean = preds.mean(dim=0)
        var  = preds.var(dim=0)
        return mean, var

    def predict_next(
        self,
        initial_state:  np.ndarray,
        action_history: deque,
        step_count:     int,
        device=None,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        单步预测（numpy 接口，供 DeepONetWorldModelEnv 使用）

        Args:
            initial_state:  (7,) — episode 起始真实状态（s_0，Branch 锚点，不递归）
            action_history: deque(maxlen=K) — 过去 K 步动作历史
            step_count:     int  — 当前步数（0-indexed，时间归一化用）

        Returns:
            next_state_mean: (7,)   — 均值预测下一步状态
            next_state_std:  (7,)   — 标准差（不确定性）
            reward_mean:     float  — 均值奖励
            uncertainty:     float  — 全局不确定性标量
        """
        if device is None:
            device = next(self.parameters()).device

        # 构造 branch 输入：[s_0(7), a_{t-K+1}...a_t 展平] → (67,)
        action_flat = np.concatenate(list(action_history), axis=0)  # (K*3,)
        branch_in   = np.concatenate([initial_state, action_flat]).astype(np.float32)

        # 构造 trunk 输入：[t / MAX_STEPS] → (1,)
        trunk_in = np.array([step_count / MAX_STEPS], dtype=np.float32)

        self.eval()
        with torch.no_grad():
            b = torch.tensor(branch_in, dtype=torch.float32).unsqueeze(0).to(device)
            t = torch.tensor(trunk_in,  dtype=torch.float32).unsqueeze(0).to(device)
            mean, var = self(b, t)

        mean_np = mean[0].cpu().numpy()
        var_np  = var[0].cpu().numpy()

        next_state_mean = np.clip(mean_np[:STATE_DIM], 0.0, 1.0)
        next_state_std  = np.sqrt(np.clip(var_np[:STATE_DIM], 0.0, 1.0))
        reward_mean     = float(mean_np[STATE_DIM])
        uncertainty     = float(var_np.mean())

        return next_state_mean, next_state_std, reward_mean, uncertainty


# ─────────────────────────────────────────────────────────────────────────────
# 数据构建
# ─────────────────────────────────────────────────────────────────────────────

def build_episode_dataset(
    episodes: List[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]],
    K: int       = K,
    max_steps: int = MAX_STEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 episode 列表构建 DeepONet 训练数据集（episode 级滑动窗口）。

    Args:
        episodes: 列表，每个元素是一个 episode（步序列）
                  每步：(state, action, reward, next_state)
        K:        动作历史长度（不足时补零）
        max_steps: 最大步数（时间归一化分母）

    Returns:
        branch_data: (N, 7 + 3*K) — branch 输入（[s_0, 动作历史展平]）
        trunk_data:  (N, 1)        — trunk 输入（归一化时间步）
        target_data: (N, 8)        — 目标（[next_state(7), reward(1)]）
    """
    branch_list, trunk_list, target_list = [], [], []
    zero_action = np.zeros(ACTION_DIM, dtype=np.float32)

    for ep in episodes:
        if len(ep) == 0:
            continue
        # s_0：episode 起始状态，作为 Branch 锚点（整个 episode 不变）
        s_0 = ep[0][0].astype(np.float32)
        action_hist = deque([zero_action.copy() for _ in range(K)], maxlen=K)

        for t_step, (state, action, reward, next_state) in enumerate(ep):
            action = action.astype(np.float32)
            action_hist.append(action)

            # Branch 输入：[s_0(7), a_{t-K+1}...a_t] 展平 → (67,)
            action_flat = np.concatenate(list(action_hist), axis=0)  # (K*3,)
            branch = np.concatenate([s_0, action_flat]).astype(np.float32)

            # Trunk 输入：归一化时间步 → (1,)
            trunk = np.array([(t_step + 1) / max_steps], dtype=np.float32)

            # 目标：[next_state(7), reward(1)] → (8,)
            target = np.concatenate([
                next_state.astype(np.float32),
                np.array([float(reward)], dtype=np.float32),
            ])

            branch_list.append(branch)
            trunk_list.append(trunk)
            target_list.append(target)

    return (
        np.array(branch_list, dtype=np.float32),
        np.array(trunk_list,  dtype=np.float32),
        np.array(target_list, dtype=np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 训练入口
# ─────────────────────────────────────────────────────────────────────────────

def train_deeponet_world_model(
    episodes:   List[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]],
    n_epochs:   int   = 200,
    lr:         float = 1e-3,
    batch_size: int   = 512,
    n_models:   int   = 3,
    val_ratio:  float = 0.1,
) -> "FusionDeepONetEnsemble":
    """
    训练 FusionDeepONetEnsemble。

    Args:
        episodes:   episode 列表，每步 (state, action, reward, next_state)
        n_epochs:   训练轮数（目标 val_loss < 0.01）
        lr:         学习率
        batch_size: 批大小
        n_models:   Ensemble 数量（3 个 DeepONet）
        val_ratio:  验证集比例

    Returns:
        训练好的 FusionDeepONetEnsemble（已加载最优权重）
    """
    print(f"\n[DeepONet] 构建训练数据集（{len(episodes)} episodes）...")
    branch_data, trunk_data, target_data = build_episode_dataset(episodes)
    N = len(branch_data)
    print(f"[DeepONet] 数据集大小：{N} 样本（目标 ≥ 60K）")

    # 奖励归一化（统一到 [0, 1]）
    rew_min = float(target_data[:, -1].min())
    rew_max = float(target_data[:, -1].max())
    target_data[:, -1] = (target_data[:, -1] - rew_min) / (rew_max - rew_min + 1e-8)
    print(f"[DeepONet] 奖励归一化：[{rew_min:.2f}, {rew_max:.2f}] → [0, 1]")

    # Train/Val split
    n_val   = max(1, int(N * val_ratio))
    n_train = N - n_val
    perm    = np.random.permutation(N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    b_train = torch.tensor(branch_data[train_idx], dtype=torch.float32)
    t_train = torch.tensor(trunk_data[train_idx],  dtype=torch.float32)
    y_train = torch.tensor(target_data[train_idx], dtype=torch.float32)
    b_val   = torch.tensor(branch_data[val_idx],   dtype=torch.float32)
    t_val   = torch.tensor(trunk_data[val_idx],    dtype=torch.float32)
    y_val   = torch.tensor(target_data[val_idx],   dtype=torch.float32)

    train_ds     = TensorDataset(b_train, t_train, y_train)
    val_ds       = TensorDataset(b_val,   t_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"[DeepONet] 训练设备：{device} | 训练：{n_train} | 验证：{n_val}")

    ensemble   = FusionDeepONetEnsemble(n_models=n_models).to(device)
    optimizers = [optim.Adam(m.parameters(), lr=lr) for m in ensemble.models]
    criterion  = nn.MSELoss()

    best_val_loss = float("inf")
    print(f"\n[DeepONet] 开始训练 Ensemble（{n_models} 模型 × {n_epochs} epochs）\n")

    for epoch in range(1, n_epochs + 1):
        # 训练：每个 DeepONet 独立更新
        for m, opt in zip(ensemble.models, optimizers):
            m.train()
            for b_batch, t_batch, y_batch in train_loader:
                b_batch = b_batch.to(device)
                t_batch = t_batch.to(device)
                y_batch = y_batch.to(device)
                pred = m(b_batch, t_batch)
                loss = criterion(pred, y_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # 验证：ensemble 均值
        ensemble.eval()
        val_losses = []
        with torch.no_grad():
            for b_batch, t_batch, y_batch in val_loader:
                b_batch = b_batch.to(device)
                t_batch = t_batch.to(device)
                y_batch = y_batch.to(device)
                mean, _ = ensemble(b_batch, t_batch)
                val_losses.append(criterion(mean, y_batch).item())

        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            for i, m in enumerate(ensemble.models):
                torch.save(m.state_dict(), str(DON_DIR / f"fusion_deeponet_{i}.pt"))

        if epoch % 20 == 0 or epoch == n_epochs:
            status = "✅" if best_val_loss < 0.01 else "⏳"
            print(f"  Epoch {epoch:>4d}/{n_epochs} | val_loss={val_loss:.5f} | "
                  f"best={best_val_loss:.5f} {status}（目标<0.01）")

    # 加载最优权重
    for i, m in enumerate(ensemble.models):
        m.load_state_dict(
            torch.load(str(DON_DIR / f"fusion_deeponet_{i}.pt"), map_location=device)
        )
    ensemble.eval()

    # 保存配置
    config = {
        "K": K, "p": P, "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
        "n_models": n_models, "max_steps": MAX_STEPS,
        "rew_min": rew_min, "rew_max": rew_max,
    }
    with open(str(DON_DIR / "deeponet_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[DeepONet] 训练完成！最优 val_loss={best_val_loss:.5f}")
    print(f"[DeepONet] 保存路径：{DON_DIR}/fusion_deeponet_{{0..{n_models-1}}}.pt")
    verdict = "✅ PASS（< 0.01）" if best_val_loss < 0.01 else "❌ FAIL（需更多数据/轮次）"
    print(f"[DeepONet] 验收状态：{verdict}")

    return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# 加载工具
# ─────────────────────────────────────────────────────────────────────────────

def load_deeponet_ensemble(device=None) -> Optional["FusionDeepONetEnsemble"]:
    """
    加载已保存的 FusionDeepONetEnsemble。
    返回 None 若模型文件不存在。
    """
    if device is None:
        device = (torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))

    config_path = DON_DIR / "deeponet_config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        cfg = json.load(f)

    n_models = cfg.get("n_models", 3)
    ensemble = FusionDeepONetEnsemble(n_models=n_models).to(device)

    for i in range(n_models):
        pt_path = DON_DIR / f"fusion_deeponet_{i}.pt"
        if not pt_path.exists():
            return None
        ensemble.models[i].load_state_dict(
            torch.load(str(pt_path), map_location=device)
        )

    ensemble.eval()
    return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# Episode 持久化（供 WM 增量更新用）
# ─────────────────────────────────────────────────────────────────────────────

EPISODES_NPZ = DON_DIR / "phase1_episodes.npz"


def save_episodes_npz(
    episodes: List[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]],
    path: Path = EPISODES_NPZ,
) -> None:
    """把 episode 列表压缩保存，供下次 WM 增量训练时合并用。"""
    states, actions, rewards, next_states, ep_ids = [], [], [], [], []
    for ep_id, ep in enumerate(episodes):
        for s, a, r, ns in ep:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            ep_ids.append(ep_id)
    np.savez_compressed(
        str(path),
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        next_states=np.array(next_states, dtype=np.float32),
        ep_ids=np.array(ep_ids, dtype=np.int32),
    )
    print(f"[DeepONet] Episodes 已保存：{path}（{len(episodes)} eps, {len(states)} steps）")


def load_episodes_npz(
    path: Path = EPISODES_NPZ,
) -> List[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]]:
    """从 npz 还原 episode 列表格式。"""
    if not Path(path).exists():
        return []
    d = np.load(str(path))
    ep_ids = d["ep_ids"]
    episodes: List = []
    for ep_id in np.unique(ep_ids):
        mask = ep_ids == ep_id
        ep = list(zip(
            d["states"][mask],
            d["actions"][mask],
            d["rewards"][mask].tolist(),
            d["next_states"][mask],
        ))
        episodes.append(ep)
    print(f"[DeepONet] Episodes 已加载：{path}（{len(episodes)} eps）")
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# World Model 增量更新
# ─────────────────────────────────────────────────────────────────────────────

def finetune_deeponet_world_model(
    new_episodes: List[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]],
    old_episodes_path: Path = EPISODES_NPZ,
    n_epochs: int   = 50,
    lr:       float = 1e-4,
    batch_size: int = 512,
) -> "FusionDeepONetEnsemble":
    """
    在已有 DeepONet 基础上增量更新（避免灾难性遗忘）。

    策略：合并旧 Phase 1 episodes + 新 Phase 4 episodes，用低学习率继续训练。
    旧 episodes 提供稳定基础，新 episodes 让 WM 认识高水平策略的轨迹。

    Args:
        new_episodes:      新轨迹（Phase 4 真实环境 episodes）
        old_episodes_path: 旧 Phase 1 episodes 路径（防灾难性遗忘）
        n_epochs:          微调轮数（默认 50，远少于全量训练的 200）
        lr:                微调学习率（默认 1e-4，比初训 1e-3 低 10x）
    """
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    print("\n[WM 增量更新] 加载已有 DeepONet 权重...")
    ensemble = load_deeponet_ensemble(device)
    if ensemble is None:
        print("  ⚠️ 未找到已有 WM，退回全量训练")
        return train_deeponet_world_model(new_episodes, n_epochs=200, lr=1e-3,
                                          batch_size=batch_size)

    # 合并数据：旧 Phase 1 + 新 Phase 4
    old_episodes = load_episodes_npz(old_episodes_path)
    all_episodes = old_episodes + new_episodes
    print(f"[WM 增量更新] 数据：旧 {len(old_episodes)} eps + 新 {len(new_episodes)} eps "
          f"= 合计 {len(all_episodes)} eps")

    branch_data, trunk_data, target_data = build_episode_dataset(all_episodes)
    N = len(branch_data)

    # 奖励归一化
    rew_min = float(target_data[:, -1].min())
    rew_max = float(target_data[:, -1].max())
    target_data[:, -1] = (target_data[:, -1] - rew_min) / (rew_max - rew_min + 1e-8)

    n_val   = max(1, int(N * 0.1))
    n_train = N - n_val
    perm    = np.random.permutation(N)

    b_train = torch.tensor(branch_data[perm[:n_train]], dtype=torch.float32)
    t_train = torch.tensor(trunk_data[perm[:n_train]],  dtype=torch.float32)
    y_train = torch.tensor(target_data[perm[:n_train]], dtype=torch.float32)
    b_val   = torch.tensor(branch_data[perm[n_train:]], dtype=torch.float32)
    t_val   = torch.tensor(trunk_data[perm[n_train:]],  dtype=torch.float32)
    y_val   = torch.tensor(target_data[perm[n_train:]], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(b_train, t_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(b_val, t_val, y_val),
                              batch_size=batch_size)

    ensemble = ensemble.to(device)
    optimizers = [optim.Adam(m.parameters(), lr=lr) for m in ensemble.models]
    criterion  = nn.MSELoss()

    print(f"[WM 增量更新] 微调 {n_epochs} epochs（lr={lr}）| 训练:{n_train} 验证:{n_val}\n")
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        for m, opt in zip(ensemble.models, optimizers):
            m.train()
            for b_b, t_b, y_b in train_loader:
                pred = m(b_b.to(device), t_b.to(device))
                loss = criterion(pred, y_b.to(device))
                opt.zero_grad(); loss.backward(); opt.step()

        ensemble.eval()
        val_losses = []
        with torch.no_grad():
            for b_b, t_b, y_b in val_loader:
                mean, _ = ensemble(b_b.to(device), t_b.to(device))
                val_losses.append(criterion(mean, y_b.to(device)).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            for i, m in enumerate(ensemble.models):
                torch.save(m.state_dict(), str(DON_DIR / f"fusion_deeponet_{i}.pt"))

        if epoch % 10 == 0 or epoch == n_epochs:
            status = "✅" if best_val_loss < 0.01 else "⏳"
            print(f"  Epoch {epoch:>3d}/{n_epochs} | val_loss={val_loss:.5f} | "
                  f"best={best_val_loss:.5f} {status}")

    # 更新配置中的奖励归一化范围
    cfg_path = DON_DIR / "deeponet_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["rew_min"] = rew_min
        cfg["rew_max"] = rew_max
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    for i, m in enumerate(ensemble.models):
        m.load_state_dict(torch.load(str(DON_DIR / f"fusion_deeponet_{i}.pt"),
                                     map_location=device))
    ensemble.eval()
    print(f"\n[WM 增量更新] 完成！best_val_loss={best_val_loss:.5f}")
    return ensemble
