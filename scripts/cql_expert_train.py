"""
cql_expert_train.py — Phase 3 · Offline-RL（CQL 专家轨迹版）

【阶段定位】
  Phase 3 · Offline-RL：历史数据离线强化学习（CQL）
  ← 数据来源：Phase 1 · Sim-RL（默认 p1_ppo_v7，回退 v5）生成的专家轨迹
  ← 可选：Phase 4 精调轨迹（extra_buffer）回流，丰富数据集
  → 输出：p3_cql_v2.d3 / p3_cql_v3.d3（可通过参数指定）

【pipeline v2 中的角色】
  Stage 2：source=p1_ppo_v7/final.zip，save=p3_cql_v2.d3
  Stage 5：source=p1_ppo_v7/final.zip，extra_buffer=p4_finetune_buffer.npz，save=p3_cql_v3.d3

运行：
    venv/bin/python scripts/cql_expert_train.py [选项]

选项：
    --source_model   PPO 模型路径（默认 p1_ppo_v7，回退 v5）
    --save_dir       输出目录（默认 data/rl_models/p3_cql）
    --save_name      输出文件名（默认 p3_cql_v2.d3）
    --extra_buffer   额外 npz 轨迹路径（Phase 4 回流数据，可选）
    --n_episodes     专家 episode 数量（默认 150）
    --n_steps        CQL 训练步数（默认 200000）
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from backend.rl.fusion_env import FusionEnv
import d3rlpy

# ─── 默认路径 ────────────────────────────────────────────────────────────────
P1_V7_PATH = "data/rl_models/p1_ppo_v7/final.zip"
P1_V5_PATH = "data/rl_models/ppo_fusion_v5/final.zip"


def collect_expert_episodes(ppo_model, n_episodes: int):
    """用 PPO 模型在 FusionEnv 收集专家轨迹，返回 d3rlpy 格式的 numpy 数组。"""
    env = FusionEnv(max_steps=500)
    observations, actions, rewards, next_observations, terminals, timeouts = [], [], [], [], [], []
    lawson_count, disruption_count = 0, 0

    for ep_idx in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_steps = []

        while not done:
            action, _ = ppo_model.predict(obs, deterministic=False)
            next_obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_steps.append({
                "obs": obs.copy(), "action": action.copy(), "reward": float(rew),
                "next_obs": next_obs.copy(), "terminated": bool(terminated), "truncated": bool(truncated),
            })
            obs = next_obs

        for i, step in enumerate(ep_steps):
            is_last = (i == len(ep_steps) - 1)
            observations.append(step["obs"])
            actions.append(step["action"])
            rewards.append(step["reward"])
            next_observations.append(step["next_obs"])
            terminals.append(1.0 if (is_last and step["terminated"]) else 0.0)
            timeouts.append(1.0 if (is_last and step["truncated"]) else 0.0)

        if info.get("lawson_achieved"):
            lawson_count += 1
        if info.get("disrupted"):
            disruption_count += 1

        if (ep_idx + 1) % 30 == 0:
            print(f"  episode {ep_idx+1:3d}/{n_episodes} | "
                  f"lawson={lawson_count}/{ep_idx+1} | "
                  f"disrupt={disruption_count}/{ep_idx+1}")

    return (
        np.array(observations,      dtype=np.float32),
        np.array(actions,           dtype=np.float32),
        np.array(rewards,           dtype=np.float32),
        np.array(next_observations, dtype=np.float32),
        np.array(terminals,         dtype=np.float32),
        np.array(timeouts,          dtype=np.float32),
        lawson_count, disruption_count,
    )


def merge_extra_buffer(obs_arr, act_arr, rew_arr, nobs_arr, term_arr, tout_arr,
                       extra_buffer_path: str):
    """将 Phase 4 精调轨迹（npz）合并到现有数据集。"""
    extra = np.load(extra_buffer_path)
    print(f"  合并 Phase 4 轨迹：{extra_buffer_path}")
    print(f"  额外数据量：{len(extra['observations'])} transitions")

    obs_arr  = np.concatenate([obs_arr,  extra["observations"]],      axis=0)
    act_arr  = np.concatenate([act_arr,  extra["actions"]],           axis=0)
    rew_arr  = np.concatenate([rew_arr,  extra["rewards"]],           axis=0)
    nobs_arr = np.concatenate([nobs_arr, extra["next_observations"]], axis=0)
    term_arr = np.concatenate([term_arr, extra["terminals"]],         axis=0)
    tout_arr = np.concatenate([tout_arr, extra.get("timeouts",
                               np.zeros(len(extra["terminals"]), dtype=np.float32))], axis=0)

    print(f"  合并后总计：{len(obs_arr)} transitions")
    return obs_arr, act_arr, rew_arr, nobs_arr, term_arr, tout_arr


def main(source_model: str, save_dir: str, save_name: str,
         extra_buffer: str, n_episodes: int, n_steps: int):

    print("=" * 60)
    print(f"Phase 3 · Offline-RL — CQL + 专家轨迹 ({save_name})")
    print("=" * 60)
    print("\n【指标说明】")
    print("  critic_loss       — Q 函数 TD 误差（越小越稳定）")
    print("  conservative_loss — CQL 覆盖惩罚（防止 OOD 动作 Q 过高）")
    print("  actor_loss        — Actor 策略损失（应逐渐减小）")
    print("  temp              — SAC 熵系数（固定 0.1，防发散）\n")

    # ─── 1. 加载 PPO 模型（优先 best.zip → final.zip → fallback v5）────────────
    # ★ best.zip 是训练中 mean_reward 最高时的检查点，避免用训练末期崩塌的 final.zip
    best_candidate = str(Path(source_model).parent / "best.zip")
    if Path(best_candidate).exists():
        source_model = best_candidate
        print(f"[1/4] 加载 PPO 模型（best checkpoint）：{source_model}")
    else:
        print(f"[1/4] 加载 PPO 模型：{source_model}")
    if not Path(source_model).exists():
        print(f"  ⚠️ {source_model} 不存在，回退到 v5：{P1_V5_PATH}")
        source_model = P1_V5_PATH
        if not Path(source_model).exists():
            raise FileNotFoundError(f"v5 模型也不存在：{P1_V5_PATH}")
    ppo_model = PPO.load(source_model)
    print(f"  ✅ 已加载：{source_model}")

    # ─── 2. 收集专家轨迹 ──────────────────────────────────────────────────────
    print(f"\n[1/4] 收集 {n_episodes} 个专家 episodes...")
    (obs_arr, act_arr, rew_arr, nobs_arr, term_arr, tout_arr,
     lawson_count, disrupt_count) = collect_expert_episodes(ppo_model, n_episodes)

    n_trans = len(obs_arr)
    print(f"\n  专家数据集：{n_trans} transitions")
    print(f"  奖励：[{rew_arr.min():.2f}, {rew_arr.max():.2f}]，均值={rew_arr.mean():.2f}")
    print(f"  Lawson={lawson_count/n_episodes*100:.0f}%  破裂率={disrupt_count/n_episodes*100:.0f}%")

    # ─── 3. 合并 Phase 4 回流轨迹（可选）────────────────────────────────────
    if extra_buffer and Path(extra_buffer).exists():
        print(f"\n[2/4] 合并 Phase 4 精调轨迹...")
        obs_arr, act_arr, rew_arr, nobs_arr, term_arr, tout_arr = merge_extra_buffer(
            obs_arr, act_arr, rew_arr, nobs_arr, term_arr, tout_arr, extra_buffer
        )
    elif extra_buffer:
        print(f"\n[2/4] extra_buffer 路径不存在（跳过）：{extra_buffer}")
    else:
        print(f"\n[2/4] 无 extra_buffer（跳过合并）")

    # ─── 4. 构建 d3rlpy 数据集 ────────────────────────────────────────────────
    print("\n[3/4] 构建 d3rlpy 数据集...")
    # ★ 修复：Phase 4 轨迹用 dones 作 terminals（含 truncated），合并后可能与
    #   Phase 1 的 timeouts 重叠，违反 d3rlpy 断言。以 terminal 优先，清零冲突的 timeout。
    conflict_mask = (term_arr == 1.0) & (tout_arr == 1.0)
    if conflict_mask.sum() > 0:
        print(f"  ⚠️ 检测到 {conflict_mask.sum()} 个 terminal/timeout 冲突，已自动修正（terminal 优先）")
        tout_arr = tout_arr.copy()
        tout_arr[conflict_mask] = 0.0
    # ★ 奖励归一化：z-score 使 reward 均值=0 方差=1，防止混合数据集 Q 值爆炸
    rew_mean = float(rew_arr.mean())
    rew_std  = float(rew_arr.std()) + 1e-8
    rew_arr_norm = ((rew_arr - rew_mean) / rew_std).astype(np.float32)
    print(f"  奖励归一化：原始 [{rew_arr.min():.2f}, {rew_arr.max():.2f}] "
          f"均值={rew_mean:.2f} → 归一化后 [{rew_arr_norm.min():.3f}, {rew_arr_norm.max():.3f}]")

    dataset = d3rlpy.dataset.MDPDataset(
        observations=obs_arr,
        actions=act_arr,
        rewards=rew_arr_norm,
        terminals=term_arr,
        timeouts=tout_arr,
    )
    print(f"  数据集：{len(obs_arr)} transitions | "
          f"terminal={int(term_arr.sum())} | timeout={int(tout_arr.sum())}")

    # ─── 5. 训练 CQL ─────────────────────────────────────────────────────────
    print(f"\n[3/4] 开始 CQL 训练（{n_steps} steps，固定温度）...")
    # ★ 奖励归一化后 Q 值量级 ~[-100, 200]，conservative_weight 需对应降低
    # ★ 原始奖励（~179/step）→ 归一化后（~1.0/step），保守惩罚 5.0 太强会让 actor 极度悲观
    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=0.0,          # alpha 冻结
        temp_learning_rate=0.0,           # 温度冻结
        initial_temperature=0.1,
        batch_size=256,
        conservative_weight=1.0,          # ★ 降低（5.0→1.0），匹配归一化奖励量级
        n_critics=2,
        actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=[256, 256]),
        critic_encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=[256, 256]),
        actor_optim_factory=d3rlpy.optimizers.AdamFactory(clip_grad_norm=10.0),   # ★ 梯度裁剪
        critic_optim_factory=d3rlpy.optimizers.AdamFactory(clip_grad_norm=10.0),  # ★ 梯度裁剪
    # ★ 强制 CPU：d3rlpy 在 MPS 上 epoch 结束时会 silent crash（Apple Silicon）
    ).create(device='cpu')

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    cql_save = str(save_dir_path / save_name)

    # ★ FileAdapter 替代 Noop：每 epoch 写入日志，方便排查崩溃位置
    log_dir = save_dir_path / "cql_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cql.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=10_000,
        save_interval=20,
        evaluators={},
        logger_adapter=d3rlpy.logging.FileAdapterFactory(root_dir=str(log_dir)),
    )

    print(f"\n[3/4] CQL 训练完成，保存：{cql_save}")
    cql.save(cql_save)

    # ─── 6. 评估（100 episodes）────────────────────────────────────────────
    print("\n[4/4] FusionEnv 100-episode 评估...")
    eval_env = FusionEnv(max_steps=500)
    rewards_eval, lawsons_eval, disrupts_eval = [], [], []

    for ep_idx in range(100):
        obs, _ = eval_env.reset()
        done = False; ep_rew = 0.0; ep_law = False; ep_dis = False
        while not done:
            action = cql.predict(obs.reshape(1, -1))[0]
            obs, rew, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_rew += rew
            if info.get("lawson_achieved"): ep_law = True
            if info.get("disrupted"):       ep_dis = True
        rewards_eval.append(ep_rew)
        lawsons_eval.append(ep_law)
        disrupts_eval.append(ep_dis)
        if (ep_idx + 1) % 20 == 0:
            print(f"  ep {ep_idx+1:3d}/100 | mean_rew={np.mean(rewards_eval):.1f} | "
                  f"lawson={sum(lawsons_eval)}/{ep_idx+1}")

    mean_rew = float(np.mean(rewards_eval))
    threshold = 88_294  # Phase 1 v5 基线的 50%
    status    = "✅ PASS" if mean_rew > threshold else "❌ FAIL"

    print("\n" + "=" * 60)
    print(f"Phase 3 CQL 评估结果（{save_name}）")
    print("=" * 60)
    print(f"  100-episode mean_reward : {mean_rew:.1f}")
    print(f"  破裂率                  : {np.mean(disrupts_eval)*100:.0f}%")
    print(f"  Lawson 达成率           : {np.mean(lawsons_eval)*100:.0f}%")
    print(f"  v5 基线                 : 176,588")
    print(f"  验收线（50%）           : {threshold:,}")
    print(f"  结论                    : {status}")
    print("=" * 60)

    return cql_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 CQL 训练（专家轨迹版）")
    parser.add_argument("--source_model",  type=str, default=P1_V7_PATH,
                        help="Phase 1 PPO 模型路径（默认 v7，不存在则回退 v5）")
    parser.add_argument("--save_dir",      type=str, default="data/rl_models/p3_cql",
                        help="输出目录")
    parser.add_argument("--save_name",     type=str, default="p3_cql_v2.d3",
                        help="输出文件名")
    parser.add_argument("--extra_buffer",  type=str, default=None,
                        help="Phase 4 精调轨迹路径（.npz，可选）")
    parser.add_argument("--n_episodes",    type=int, default=150,
                        help="专家 episode 数量")
    parser.add_argument("--n_steps",       type=int, default=200_000,
                        help="CQL 训练步数")
    args = parser.parse_args()

    main(
        source_model=args.source_model,
        save_dir=args.save_dir,
        save_name=args.save_name,
        extra_buffer=args.extra_buffer,
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
    )
