"""
mbrl_expert_train.py — Phase 4 · Model-RL（世界模型强化学习）

【阶段定位】
  Phase 4 · Model-RL：世界模型强化学习（Dyna MBRL）
  ← 世界模型数据：Phase 1 · Sim-RL 专家轨迹（高质量真实环境数据）
  ← 初始 Actor：Phase 3→4 BC 蒸馏 Actor（p3_to_p4_actor.pt）/ 回退 Phase 1 v5
  → 输出：p4_mbrl/p4_mbrl_v3_best.zip（供 phase4_finetune.py 精调）

【热启动优先级】
  1. p3_to_p4_actor.pt（BC 蒸馏，CQL 策略 → PPO 权重映射）
  2. p1_ppo_v7/final.zip（Phase 1 最新版本）
  3. ppo_fusion_v5/final.zip（最终回退）

  显式 key 映射（DistillActorMLP → SB3 PPO）：
    net.0.weight → mlp_extractor.policy_net.0.weight  [256, 7]
    net.0.bias   → mlp_extractor.policy_net.0.bias    [256]
    net.2.weight → mlp_extractor.policy_net.2.weight  [256, 256]
    net.2.bias   → mlp_extractor.policy_net.2.bias    [256]
    net.4.weight → action_net.weight                   [3, 256]
    net.4.bias   → action_net.bias                     [3]

运行：
    venv/bin/python scripts/mbrl_expert_train.py [--p3_actor PATH]
"""

import sys
import argparse
import numpy as np
import torch
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from backend.rl.fusion_env import FusionEnv
from backend.rl.world_model import WorldModelEnsemble, WM_DIR
from backend.rl.world_model_env import WorldModelEnv, DeepONetWorldModelEnv
from backend.rl.world_model_deeponet import (
    train_deeponet_world_model,
    finetune_deeponet_world_model,
    save_episodes_npz,
    load_episodes_npz,
    EPISODES_NPZ,
)
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import shutil

# ─── 默认路径 ─────────────────────────────────────────────────────────────────
DEFAULT_P3_ACTOR = "data/rl_models/p3_cql/p3_to_p4_actor.pt"
P1_V7_PATH       = "data/rl_models/p1_ppo_v7/final.zip"
P1_V5_PATH       = "data/rl_models/ppo_fusion_v5/final.zip"

# ─── 参数 ────────────────────────────────────────────────────────────────────
N_EXPERT_EP  = 200
N_WM_EPOCHS  = 100
N_DYNA_ITER  = 100
N_RL_STEPS   = 10_000
N_ENVS       = 4

# Phase 3→PPO 显式权重映射表
P3_TO_PPO_KEY_MAP = {
    "net.0.weight": "mlp_extractor.policy_net.0.weight",
    "net.0.bias":   "mlp_extractor.policy_net.0.bias",
    "net.2.weight": "mlp_extractor.policy_net.2.weight",
    "net.2.bias":   "mlp_extractor.policy_net.2.bias",
    "net.4.weight": "action_net.weight",
    "net.4.bias":   "action_net.bias",
}


def collect_expert_episodes(source_model_path: str, n_episodes: int = None):
    """
    从 PPO 模型收集专家轨迹（episode 级别，供 DeepONet 训练用）。

    Returns:
        episodes: List[List[(obs, action, reward, next_obs)]]
    """
    n_ep = n_episodes or N_EXPERT_EP
    model_path = source_model_path
    if not Path(model_path).exists():
        print(f"  ⚠️ {model_path} 不存在，回退 v5：{P1_V5_PATH}")
        model_path = P1_V5_PATH

    print(f"[DeepONet] 加载专家模型：{model_path}")
    src_model = PPO.load(model_path)
    env_collect = FusionEnv(max_steps=500)

    print(f"[DeepONet] 收集 {n_ep} 个专家 episodes（episode 级）...")
    episodes = []
    lawson_count, disrupt_count = 0, 0

    for ep_idx in range(n_ep):
        obs, _ = env_collect.reset()
        done = False
        ep_steps = []

        while not done:
            action, _ = src_model.predict(obs, deterministic=False)
            next_obs, rew, terminated, truncated, info = env_collect.step(action)
            done = terminated or truncated
            ep_steps.append((obs.copy(), action.copy(), float(rew), next_obs.copy()))
            obs = next_obs

        episodes.append(ep_steps)
        if info.get("lawson_achieved"): lawson_count += 1
        if info.get("disrupted"):       disrupt_count += 1

        if (ep_idx + 1) % 50 == 0:
            total_steps = sum(len(e) for e in episodes)
            print(f"  episode {ep_idx+1:3d}/{n_ep} | "
                  f"lawson={lawson_count}/{ep_idx+1} | disrupt={disrupt_count}/{ep_idx+1} | "
                  f"transitions={total_steps}")

    total_steps = sum(len(e) for e in episodes)
    print(f"\n[DeepONet] 专家数据：{len(episodes)} episodes，{total_steps} transitions")
    print(f"  Lawson={lawson_count/n_ep*100:.0f}%  破裂率={disrupt_count/n_ep*100:.0f}%")
    return episodes


def collect_expert_data(source_model_path: str):
    """从 PPO 模型收集专家轨迹（优先 v7，回退 v5）。"""
    model_path = source_model_path
    if not Path(model_path).exists():
        print(f"  ⚠️ {model_path} 不存在，回退 v5：{P1_V5_PATH}")
        model_path = P1_V5_PATH

    print(f"[1/4] 加载专家模型：{model_path}")
    src_model = PPO.load(model_path)
    env_collect = FusionEnv(max_steps=500)

    print(f"[1/4] 收集 {N_EXPERT_EP} 个专家 episodes...")
    obs_list, act_list, rew_list, next_obs_list = [], [], [], []
    lawson_count, disrupt_count = 0, 0

    for ep_idx in range(N_EXPERT_EP):
        obs, _ = env_collect.reset()
        done = False
        ep_steps = []

        while not done:
            action, _ = src_model.predict(obs, deterministic=False)
            next_obs, rew, terminated, truncated, info = env_collect.step(action)
            done = terminated or truncated
            ep_steps.append((obs.copy(), action.copy(), float(rew), next_obs.copy()))
            obs = next_obs

        if info.get("lawson_achieved"): lawson_count += 1
        if info.get("disrupted"):       disrupt_count += 1

        for so, sa, sr, sno in ep_steps:
            obs_list.append(so); act_list.append(sa)
            rew_list.append(sr); next_obs_list.append(sno)

        if (ep_idx + 1) % 50 == 0:
            print(f"  episode {ep_idx+1:3d}/{N_EXPERT_EP} | "
                  f"lawson={lawson_count}/{ep_idx+1} | disrupt={disrupt_count}/{ep_idx+1} | "
                  f"transitions={len(obs_list)}")

    n_trans = len(obs_list)
    print(f"\n[1/4] 专家数据集：{n_trans} transitions")
    print(f"  奖励：[{min(rew_list):.2f}, {max(rew_list):.2f}]，均值={np.mean(rew_list):.2f}")
    print(f"  Lawson={lawson_count/N_EXPERT_EP*100:.0f}%  破裂率={disrupt_count/N_EXPERT_EP*100:.0f}%")

    return obs_list, act_list, rew_list, next_obs_list


def warmstart_from_p3_actor(agent: PPO, p3_actor_path: str) -> bool:
    """
    从 Phase 3 BC 蒸馏 Actor 热启动 PPO。
    使用显式 key 映射：DistillActorMLP.net.X → PPO.policy.mlp_extractor.policy_net.X
    """
    if not Path(p3_actor_path).exists():
        return False

    try:
        bc_weights   = torch.load(p3_actor_path, map_location="cpu")
        policy_state = agent.policy.state_dict()

        mapped = 0
        for bc_key, ppo_key in P3_TO_PPO_KEY_MAP.items():
            if bc_key in bc_weights and ppo_key in policy_state:
                bc_w = bc_weights[bc_key]
                pp_w = policy_state[ppo_key]
                if bc_w.shape == pp_w.shape:
                    policy_state[ppo_key] = bc_w
                    mapped += 1
                else:
                    print(f"  ⚠️ shape 不匹配 {bc_key}: {bc_w.shape} vs {pp_w.shape}，跳过")

        agent.policy.load_state_dict(policy_state, strict=False)
        print(f"  ✅ Phase 3 蒸馏热启动成功（映射 {mapped}/6 个参数层）")
        return mapped > 0

    except Exception as e:
        print(f"  ⚠️ Phase 3 热启动失败（{e}）")
        return False


def warmstart_from_ppo(agent: PPO, ppo_path: str, vec_env) -> bool:
    """从已有 PPO zip 文件热启动（加载 policy state dict）。"""
    if not Path(ppo_path).exists():
        return False

    try:
        src_agent = PPO.load(ppo_path, env=vec_env)
        agent.policy.load_state_dict(src_agent.policy.state_dict())
        print(f"  ✅ PPO 热启动成功：{ppo_path}")
        return True
    except Exception as e:
        print(f"  ⚠️ PPO 热启动失败（{e}）")
        return False


def main(p3_actor_path: str = DEFAULT_P3_ACTOR, source_model: str = P1_V7_PATH,
         use_deeponet: bool = True, finetune_wm: bool = False,
         extra_episodes_npz: str = ""):
    SAVE_DIR = Path("data/rl_models/p4_mbrl")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    wm_label = "DeepONet（Path C）" if use_deeponet else "MLP Ensemble（原版）"
    print("=" * 60)
    print(f"Phase 4 · Model-RL v3 — MBRL + {wm_label}")
    print("=" * 60)
    print("\n【指标说明】")
    print("  wm_val_loss  — 世界模型验证损失（越小 = 预测越准，DeepONet 目标 < 0.01）")
    print("  rl_reward    — PPO 在 FusionEnv 真实奖励（目标 ≥ 193,527）")
    print("  lawson_rate  — 劳森条件达成率（核聚变点火标准）")
    print("  disrupt_rate — 破裂率（越低越好）\n")

    # ─── DeepONet 路径（Path C）────────────────────────────────────────────────
    if use_deeponet:
        print("[Path C] 使用 FusionDeepONet 世界模型（历史感知，无递归误差）\n")

        # 步骤 1：收集 episode 级专家轨迹（供 DeepONet 训练）
        episodes = collect_expert_episodes(source_model, n_episodes=N_EXPERT_EP)

        # 持久化 Phase 1 episodes（供 WM 增量更新时防灾难性遗忘）
        save_episodes_npz(episodes, EPISODES_NPZ)

        # 步骤 2：训练 / 增量更新 FusionDeepONet Ensemble
        if finetune_wm:
            # 增量更新：合并旧 Phase 1 + 新 Phase 4 轨迹，低学习率微调
            extra_eps = load_episodes_npz(extra_episodes_npz) if extra_episodes_npz else []
            new_eps   = episodes + extra_eps
            print(f"\n[2/4] WM 增量更新（旧 WM + {len(new_eps)} eps 微调）...")
            deeponet = finetune_deeponet_world_model(
                new_episodes=new_eps,
                old_episodes_path=EPISODES_NPZ,
                n_epochs=50,
                lr=1e-4,
            )
        else:
            print(f"\n[2/4] 全量训练 FusionDeepONet（目标 val_loss < 0.01）...")
            deeponet = train_deeponet_world_model(
                episodes,
                n_epochs=200,
                lr=1e-3,
                batch_size=512,
                n_models=3,
            )

        # 步骤 3：初始化 PPO（热启动）
        print(f"\n[3/4] 初始化 PPO（热启动优先级：Phase 3 蒸馏 > PPO v7 > PPO v5 > 随机）...")

        def make_wm_env():
            return DeepONetWorldModelEnv(deeponet=deeponet, max_steps=500)

        vec_env = DummyVecEnv([make_wm_env] * N_ENVS)
        vec_env = VecMonitor(vec_env)

        agent = PPO(
            "MlpPolicy", vec_env,
            learning_rate=1e-4, n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.002, clip_range=0.1,
            policy_kwargs=dict(net_arch=[256, 256]), verbose=0,
        )

        if not warmstart_from_p3_actor(agent, p3_actor_path):
            print("  Phase 3 Actor 不可用，尝试 PPO v7 热启动...")
            if not warmstart_from_ppo(agent, P1_V7_PATH, vec_env):
                print("  PPO v7 不可用，尝试 PPO v5 热启动...")
                if not warmstart_from_ppo(agent, P1_V5_PATH, vec_env):
                    print("  ⚠️ 所有热启动均失败，使用随机初始化")

        # 步骤 4：Dyna MBRL 主循环
        print(f"\n[4/4] Dyna MBRL (DeepONet) | 迭代={N_DYNA_ITER} | 每次 PPO={N_RL_STEPS} steps")
        print(f"  保存目录：{SAVE_DIR}\n")

        # 备份已有 best/final，防止新训练结果更差时覆盖历史最优
        from datetime import datetime
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for _fname in ["p4_mbrl_v3_best.zip", "p4_mbrl_v3_final.zip", "p4_mbrl_v3_finetuned.zip"]:
            _src = SAVE_DIR / _fname
            if _src.exists():
                _bak = SAVE_DIR / f"backup_{_ts}_{_fname}"
                shutil.copy2(_src, _bak)
                print(f"  📦 备份已有模型：{_fname} → backup_{_ts}_{_fname}")

        best_reward = -float("inf")
        eval_env    = FusionEnv(max_steps=500)

        for iteration in range(1, N_DYNA_ITER + 1):
            iter_start = time.time()
            agent.learn(total_timesteps=N_RL_STEPS, reset_num_timesteps=False)

            if iteration % 5 == 0 or iteration == 1:
                eval_rewards, lawson_ok, disrupt_ok = [], 0, 0
                for _ in range(5):
                    obs, _ = eval_env.reset()
                    done = False; ep_rew = 0.0; ep_law = False; ep_dis = False
                    while not done:
                        action, _ = agent.predict(obs, deterministic=True)
                        obs, rew, terminated, truncated, info = eval_env.step(action)
                        done = terminated or truncated
                        ep_rew += rew
                        if info.get("lawson_achieved"): ep_law = True
                        if info.get("disrupted"):       ep_dis = True
                    eval_rewards.append(ep_rew)
                    if ep_law: lawson_ok += 1
                    if ep_dis: disrupt_ok += 1

                mean_reward = float(np.mean(eval_rewards))
                elapsed = time.time() - iter_start
                print(f"  迭代 {iteration:>3d}/{N_DYNA_ITER} | "
                      f"FusionEnv reward={mean_reward:>10.1f} | "
                      f"lawson={lawson_ok}/5 | disrupt={disrupt_ok}/5 | "
                      f"耗时={elapsed:.1f}s")

                if mean_reward > best_reward:
                    best_reward = mean_reward
                    agent.save(str(SAVE_DIR / "p4_mbrl_v3_best.zip"))

            if iteration % 20 == 0:
                agent.save(str(SAVE_DIR / f"p4_mbrl_v3_iter{iteration}.zip"))

        agent.save(str(SAVE_DIR / "p4_mbrl_v3_final.zip"))
        print(f"\n[Dyna MBRL DeepONet] 训练完成！最优 FusionEnv reward：{best_reward:.1f}")

        # 最终正式评估
        print("\n=== 最终 100-episode 正式评估 ===")
        best_agent = PPO.load(str(SAVE_DIR / "p4_mbrl_v3_best.zip"))
        final_rewards, final_lawsons, final_disrupts, final_lens = [], [], [], []

        for ep in range(100):
            obs, _ = eval_env.reset()
            done = False; ep_rew = 0.0; ep_law = False; ep_dis = False; ep_len = 0
            while not done:
                action, _ = best_agent.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                ep_rew += rew; ep_len += 1
                if info.get("lawson_achieved"): ep_law = True
                if info.get("disrupted"):       ep_dis = True
            final_rewards.append(ep_rew)
            final_lawsons.append(ep_law)
            final_disrupts.append(ep_dis)
            final_lens.append(ep_len)
            if (ep + 1) % 25 == 0:
                print(f"  {ep+1:3d}/100 | mean_rew={np.mean(final_rewards[-25:]):.1f} | "
                      f"lawson={int(np.sum(final_lawsons[-25:]))}/25 | "
                      f"disrupt={int(np.sum(final_disrupts[-25:]))}/25")

        print(f"\n[Phase 4 DeepONet] 最终评估（100 episodes）：")
        print(f"  mean_reward  : {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
        print(f"  lawson_rate  : {np.mean(final_lawsons)*100:.1f}%")
        print(f"  disrupt_rate : {np.mean(final_disrupts)*100:.1f}%")
        print(f"  ep_len       : {np.mean(final_lens):.1f}")

        threshold = 193_527
        status = "✅ PASS" if np.mean(final_rewards) >= threshold else "❌ FAIL"
        print(f"\n  Phase 4 DeepONet 验收（≥ {threshold:,}）: {status}")
        print(f"\n  最优模型：{SAVE_DIR}/p4_mbrl_v3_best.zip")

        print("\n【名词备注】")
        print("  DeepONet    — 深度算子网络，Branch 吃动作历史，Trunk 吃时间坐标")
        print("  s_0 锚点    — episode 起始真实状态，Branch 输入不递归，无累积误差")
        print("  K=20 历史   — 过去 20 步动作历史，打破 Markov 假设")
        print("  RK4 积分    — 4 阶 Runge-Kutta，ODE 精度从 O(dt²) 提升到 O(dt⁵)")

        return str(SAVE_DIR / "p4_mbrl_v3_best.zip")

    # ─── 原版 MLP Ensemble 路径（--use_deeponet False 时执行）──────────────────
    print("[原版] 使用 MLP Ensemble 世界模型\n")

    # ─── 步骤 1：收集专家轨迹 ─────────────────────────────────────────────────
    obs_list, act_list, rew_list, next_obs_list = collect_expert_data(source_model)

    # ─── 步骤 2：训练世界模型 ─────────────────────────────────────────────────
    print(f"\n[2/4] 训练世界模型（{N_WM_EPOCHS} epochs）...")

    obs_t    = torch.tensor(np.array(obs_list),      dtype=torch.float32)
    act_t    = torch.tensor(np.array(act_list),      dtype=torch.float32)
    rew_t    = torch.tensor(np.array(rew_list),      dtype=torch.float32).unsqueeze(1)
    nobs_t   = torch.tensor(np.array(next_obs_list), dtype=torch.float32)

    inputs  = torch.cat([obs_t, act_t], dim=1)   # (N, 10)
    targets = torch.cat([nobs_t, rew_t], dim=1)  # (N, 8)

    rew_min, rew_max = float(rew_t.min()), float(rew_t.max())
    targets[:, -1] = (rew_t.squeeze(1) - rew_min) / (rew_max - rew_min + 1e-8)
    print(f"  奖励归一化：[{rew_min:.2f}, {rew_max:.2f}] → [0, 1]")

    n       = len(obs_t)
    n_val   = max(1, int(n * 0.1))
    n_train = n - n_val
    perm    = torch.randperm(n)
    train_ds = TensorDataset(inputs[perm[:n_train]], targets[perm[:n_train]])
    val_ds   = TensorDataset(inputs[perm[n_train:]], targets[perm[n_train:]])
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"  训练：{n_train} | 验证：{n_val} | 设备：{device}")

    world_model = WorldModelEnsemble(n_models=5, hidden=256).to(device)
    optimizers  = [optim.Adam(m.parameters(), lr=3e-4) for m in world_model.models]
    criterion   = nn.MSELoss()

    best_val_loss = float("inf")
    wm_save_path  = str(WM_DIR / "world_model_v3.pt")

    for epoch in range(1, N_WM_EPOCHS + 1):
        for m, opt in zip(world_model.models, optimizers):
            m.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(m(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()

        world_model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                sb, ab = xb[:, :7], xb[:, 7:]
                mean, _ = world_model(sb, ab)
                val_losses.append(criterion(mean, yb).item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(world_model.state_dict(), wm_save_path)

        if epoch % 20 == 0 or epoch == N_WM_EPOCHS:
            print(f"  Epoch {epoch:>4d}/{N_WM_EPOCHS} | val_loss={val_loss:.5f} | best={best_val_loss:.5f}")

    world_model.load_state_dict(torch.load(wm_save_path, map_location=device))
    world_model.eval()
    pred_err = best_val_loss ** 0.5 * 100
    print(f"\n[2/4] 世界模型完成！val_loss={best_val_loss:.5f}，预测误差≈{pred_err:.1f}%（目标<5%）")

    # ─── 步骤 3：初始化 PPO（热启动优先级：p3 actor > v7 > v5 > 随机）────────
    print(f"\n[3/4] 初始化 PPO（热启动优先级：Phase 3 蒸馏 > PPO v7 > PPO v5 > 随机）...")

    import backend.rl.world_model as wm_module
    wm_module._world_model = world_model

    def make_wm_env():
        return WorldModelEnv(world_model=world_model, max_steps=500)

    vec_env = DummyVecEnv([make_wm_env] * N_ENVS)
    vec_env = VecMonitor(vec_env)

    agent = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.002,
        clip_range=0.1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
    )

    # 热启动（优先 Phase 3 蒸馏 Actor）
    if not warmstart_from_p3_actor(agent, p3_actor_path):
        print(f"  Phase 3 Actor 不可用，尝试 PPO v7 热启动...")
        if not warmstart_from_ppo(agent, P1_V7_PATH, vec_env):
            print(f"  PPO v7 不可用，尝试 PPO v5 热启动...")
            if not warmstart_from_ppo(agent, P1_V5_PATH, vec_env):
                print("  ⚠️ 所有热启动均失败，使用随机初始化")

    # ─── 步骤 4：Dyna MBRL 主循环 ─────────────────────────────────────────────
    print(f"\n[4/4] Dyna MBRL | 迭代={N_DYNA_ITER} | 每次 PPO={N_RL_STEPS} steps")
    print(f"  保存目录：{SAVE_DIR}\n")

    best_reward = -float("inf")
    eval_env    = FusionEnv(max_steps=500)

    for iteration in range(1, N_DYNA_ITER + 1):
        iter_start = time.time()
        agent.learn(total_timesteps=N_RL_STEPS, reset_num_timesteps=False)

        if iteration % 5 == 0 or iteration == 1:
            eval_rewards, lawson_ok, disrupt_ok = [], 0, 0
            for _ in range(5):
                obs, _ = eval_env.reset()
                done = False; ep_rew = 0.0; ep_law = False; ep_dis = False
                while not done:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, rew, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    ep_rew += rew
                    if info.get("lawson_achieved"): ep_law = True
                    if info.get("disrupted"):       ep_dis = True
                eval_rewards.append(ep_rew)
                if ep_law: lawson_ok += 1
                if ep_dis: disrupt_ok += 1

            mean_reward = float(np.mean(eval_rewards))
            elapsed = time.time() - iter_start

            print(f"  迭代 {iteration:>3d}/{N_DYNA_ITER} | "
                  f"FusionEnv reward={mean_reward:>10.1f} | "
                  f"lawson={lawson_ok}/5 | disrupt={disrupt_ok}/5 | "
                  f"耗时={elapsed:.1f}s")

            if mean_reward > best_reward:
                best_reward = mean_reward
                agent.save(str(SAVE_DIR / "p4_mbrl_v3_best.zip"))

        if iteration % 20 == 0:
            agent.save(str(SAVE_DIR / f"p4_mbrl_v3_iter{iteration}.zip"))

    agent.save(str(SAVE_DIR / "p4_mbrl_v3_final.zip"))
    print(f"\n[Dyna MBRL v3] 训练完成！最优 FusionEnv reward：{best_reward:.1f}")

    # ─── 最终 100-episode 正式评估 ───────────────────────────────────────────
    print("\n=== 最终 100-episode 正式评估 ===")
    best_agent  = PPO.load(str(SAVE_DIR / "p4_mbrl_v3_best.zip"))
    final_rewards, final_lawsons, final_disrupts, final_lens = [], [], [], []

    for ep in range(100):
        obs, _ = eval_env.reset()
        done = False; ep_rew = 0.0; ep_law = False; ep_dis = False; ep_len = 0
        while not done:
            action, _ = best_agent.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_rew += rew; ep_len += 1
            if info.get("lawson_achieved"): ep_law = True
            if info.get("disrupted"):       ep_dis = True
        final_rewards.append(ep_rew)
        final_lawsons.append(ep_law)
        final_disrupts.append(ep_dis)
        final_lens.append(ep_len)

        if (ep + 1) % 25 == 0:
            print(f"  {ep+1:3d}/100 | mean_rew={np.mean(final_rewards[-25:]):.1f} | "
                  f"lawson={int(np.sum(final_lawsons[-25:]))}/25 | "
                  f"disrupt={int(np.sum(final_disrupts[-25:]))}/25")

    print(f"\n[Phase 4 v3] 最终评估（100 episodes）：")
    print(f"  mean_reward  : {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
    print(f"  lawson_rate  : {np.mean(final_lawsons)*100:.1f}%")
    print(f"  disrupt_rate : {np.mean(final_disrupts)*100:.1f}%")
    print(f"  ep_len       : {np.mean(final_lens):.1f}")

    threshold = 176_588
    status = "✅ PASS" if np.mean(final_rewards) >= threshold else "❌ FAIL"
    print(f"\n  Phase 4 验收（≥ {threshold:,}）: {status}")
    print(f"\n  最优模型：{SAVE_DIR}/p4_mbrl_v3_best.zip")

    print("\n【名词备注】")
    print("  Dyna MBRL   — 世界模型辅助 RL，在虚拟环境中额外训练，提高样本效率")
    print("  BC 蒸馏热启动 — 将 CQL 策略蒸馏为 PPO 兼容 MLP，显式 key 映射权重")
    print("  Ensemble     — 多个世界模型集成，方差估计不确定性，防止 Model Bias")

    return str(SAVE_DIR / "p4_mbrl_v3_best.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 MBRL v3 训练")
    parser.add_argument("--p3_actor",     type=str, default=DEFAULT_P3_ACTOR,
                        help="Phase 3 BC 蒸馏 Actor 路径（.pt）")
    parser.add_argument("--source_model", type=str, default=P1_V7_PATH,
                        help="Phase 1 专家模型路径（用于收集世界模型数据）")
    parser.add_argument("--use_deeponet", type=lambda x: x.lower() != "false",
                        default=True,
                        help="使用 FusionDeepONet 世界模型（默认 True，--use_deeponet False 回退 MLP Ensemble）")
    parser.add_argument("--finetune_wm", action="store_true",
                        help="增量更新 World Model（而非从头训练），用于飞轮第 2 轮起")
    parser.add_argument("--extra_episodes_npz", type=str, default="",
                        help="Phase 4 真实轨迹 npz 路径（finetune_wm 时合并用）")
    args = parser.parse_args()

    main(p3_actor_path=args.p3_actor, source_model=args.source_model,
         use_deeponet=args.use_deeponet, finetune_wm=args.finetune_wm,
         extra_episodes_npz=args.extra_episodes_npz)
