"""
p3_to_p4_distill.py — Phase 3→4 BC 蒸馏

【功能】
  将 Phase 3 CQL（d3rlpy 格式）蒸馏为 SB3 PPO 兼容的 BC Actor
  ← 输入：p3_cql_v2.d3（d3rlpy CQL 策略）
  → 输出：p3_to_p4_actor.pt（用于热启动 Phase 4 PPO）

【为何需要蒸馏】
  CQL 是 d3rlpy 格式，无法直接热启动 SB3 PPO。
  通过行为克隆（BC），让一个与 Phase 4 PPO net_arch=[256,256] 对齐的 MLP
  模仿 CQL 策略，再将 MLP 权重显式映射到 PPO policy 网络，实现跨框架热启动。

  权重映射关系（mbrl_expert_train.py 使用）：
    net.0.weight → mlp_extractor.policy_net.0.weight  [256, 7]
    net.0.bias   → mlp_extractor.policy_net.0.bias    [256]
    net.2.weight → mlp_extractor.policy_net.2.weight  [256, 256]
    net.2.bias   → mlp_extractor.policy_net.2.bias    [256]
    net.4.weight → action_net.weight                   [3, 256]
    net.4.bias   → action_net.bias                     [3]

运行：
    venv/bin/python scripts/p3_to_p4_distill.py [--cql_path PATH] [--save_path PATH]
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import d3rlpy
from backend.rl.fusion_env import FusionEnv

# ─── 默认路径 ──────────────────────────────────────────────────────────────────
DEFAULT_CQL_PATH  = "data/rl_models/p3_cql/p3_cql_v2.d3"
DEFAULT_SAVE_PATH = "data/rl_models/p3_cql/p3_to_p4_actor.pt"
N_DISTILL_EP = 150
BC_EPOCHS    = 500
BC_LR        = 1e-3
BATCH_SIZE   = 256
STATE_DIM    = 7
ACTION_DIM   = 3
HIDDEN       = 256   # 与 Phase 4 PPO net_arch=[256,256] 对齐


class DistillActorMLP(nn.Module):
    """
    蒸馏 Actor — net_arch=[256,256]，与 Phase 4 PPO policy_net 层数/维度完全对齐。

    Sequential 中层索引对应关系：
      net.0 = Linear(7→256)   → 映射到 mlp_extractor.policy_net.0
      net.1 = ReLU()           （无参数，跳过）
      net.2 = Linear(256→256) → 映射到 mlp_extractor.policy_net.2
      net.3 = ReLU()           （无参数，跳过）
      net.4 = Linear(256→3)   → 映射到 action_net
      net.5 = Tanh()           （无参数，跳过）
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN),   # net.0
            nn.ReLU(),                       # net.1
            nn.Linear(HIDDEN, HIDDEN),      # net.2
            nn.ReLU(),                       # net.3
            nn.Linear(HIDDEN, ACTION_DIM),  # net.4
            nn.Tanh(),                       # net.5
        )
        self._scale = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self._scale


def main(cql_path: str, save_path: str, n_distill_ep: int = N_DISTILL_EP,
         bc_epochs: int = BC_EPOCHS):
    print("=" * 60)
    print("Phase 3→4 BC 蒸馏 — CQL 策略 → SB3 PPO Actor")
    print("=" * 60)
    print("\n【指标说明】")
    print("  n_transitions — 蒸馏收集的 (obs, action) 对数量")
    print("  train_loss    — BC 拟合损失（MSE，越小 = 蒸馏越准）")
    print("  val_loss      — 验证损失（衡量泛化性，目标 < 1e-4）\n")

    # ─── 1. 加载 CQL ────────────────────────────────────────────────────────
    print(f"[1/3] 加载 CQL：{cql_path}")
    if not Path(cql_path).exists():
        raise FileNotFoundError(f"CQL 模型不存在：{cql_path}")
    cql = d3rlpy.load_learnable(cql_path)
    print("  CQL 加载成功")

    # ─── 2. 收集蒸馏数据（CQL 在 FusionEnv 跑 N 个 episode）────────────────
    print(f"\n[2/3] 用 CQL 跑 {n_distill_ep} episodes 收集蒸馏数据...")
    env = FusionEnv(max_steps=500)
    obs_buf, act_buf = [], []
    lawson_count, disrupt_count = 0, 0

    for ep_idx in range(n_distill_ep):
        obs, _ = env.reset()
        done = False
        ep_obs, ep_act = [], []

        while not done:
            action = cql.predict(obs.reshape(1, -1))[0]
            next_obs, rew, terminated, truncated, info = env.step(action)
            ep_obs.append(obs.copy())
            ep_act.append(action.copy())
            obs = next_obs
            done = terminated or truncated

        obs_buf.extend(ep_obs)
        act_buf.extend(ep_act)

        if info.get("lawson_achieved"):
            lawson_count += 1
        if info.get("disrupted"):
            disrupt_count += 1

        if (ep_idx + 1) % 30 == 0:
            print(f"  episode {ep_idx+1:3d}/{n_distill_ep} | "
                  f"lawson={lawson_count}/{ep_idx+1} | "
                  f"disrupt={disrupt_count}/{ep_idx+1} | "
                  f"transitions={len(obs_buf)}")

    obs_arr = np.array(obs_buf, dtype=np.float32)
    act_arr = np.array(act_buf, dtype=np.float32)
    n_trans = len(obs_arr)
    print(f"\n  蒸馏数据集：{n_trans} transitions")
    print(f"  动作范围：[{act_arr.min():.4f}, {act_arr.max():.4f}]")
    print(f"  Lawson达成：{lawson_count / n_distill_ep * 100:.0f}%  "
          f"破裂率：{disrupt_count / n_distill_ep * 100:.0f}%")

    # ─── 3. BC 训练（MSE 拟合 CQL 动作）────────────────────────────────────
    print(f"\n[3/3] BC 蒸馏训练（{bc_epochs} epochs，hidden={HIDDEN}）...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"  训练设备：{device}")

    obs_t = torch.tensor(obs_arr, dtype=torch.float32)
    act_t = torch.tensor(act_arr, dtype=torch.float32)

    n_val   = max(1, int(n_trans * 0.1))
    n_train = n_trans - n_val
    perm    = torch.randperm(n_trans)
    train_obs = obs_t[perm[:n_train]]
    train_act = act_t[perm[:n_train]]
    val_obs   = obs_t[perm[n_train:]]
    val_act   = act_t[perm[n_train:]]

    actor     = DistillActorMLP().to(device)
    optimizer = optim.Adam(actor.parameters(), lr=BC_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=bc_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, bc_epochs + 1):
        actor.train()
        perm_train  = torch.randperm(n_train)
        batch_losses = []
        for start in range(0, n_train, BATCH_SIZE):
            idx  = perm_train[start : start + BATCH_SIZE]
            xb   = train_obs[idx].to(device)
            yb   = train_act[idx].to(device)
            loss = criterion(actor(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        actor.eval()
        with torch.no_grad():
            val_loss = criterion(actor(val_obs.to(device)), val_act.to(device)).item()

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(actor.state_dict(), save_path)

        if epoch % 100 == 0 or epoch == bc_epochs:
            print(f"  Epoch {epoch:>4d}/{bc_epochs} | "
                  f"train={np.mean(batch_losses):.6f} | "
                  f"val={val_loss:.6f} | best_val={best_val_loss:.6f}")

    verdict = "✅ 蒸馏效果良好" if best_val_loss < 1e-4 else "⚠️ val_loss 偏高，热启动效果有限"
    print(f"\n  最优 val_loss={best_val_loss:.6f}  {verdict}")
    print(f"  蒸馏 Actor 已保存：{save_path}")

    print("\n【名词备注】")
    print("  BC 蒸馏 — 行为克隆蒸馏，让小网络模仿 CQL 策略动作（MSE 监督学习）")
    print("  — what: 用 CQL 推理结果作为标签，训练一个 SB3 PPO 兼容的 MLP")
    print("  — why:  CQL 是 d3rlpy 格式，无法直接热启动 SB3 PPO，需格式转换")
    print("  — how:  val_loss < 1e-4 时蒸馏效果较好；网络结构与 Phase 4 PPO 完全对齐")

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3→4 BC 蒸馏")
    parser.add_argument("--cql_path",    type=str, default=DEFAULT_CQL_PATH,
                        help="CQL 模型路径（.d3 格式）")
    parser.add_argument("--save_path",   type=str, default=DEFAULT_SAVE_PATH,
                        help="蒸馏 Actor 保存路径（.pt）")
    parser.add_argument("--n_distill_ep", type=int, default=N_DISTILL_EP,
                        help="蒸馏 episodes 数量")
    parser.add_argument("--bc_epochs",   type=int, default=BC_EPOCHS,
                        help="BC 训练轮次")
    args = parser.parse_args()

    main(
        cql_path=args.cql_path,
        save_path=args.save_path,
        n_distill_ep=args.n_distill_ep,
        bc_epochs=args.bc_epochs,
    )
