"""
eval_phase4_mbrl.py — Phase 4 · Model-RL 100-episode 正式评估

【用途】
  使用真实 FusionEnv（非 WorldModelEnv）对 Phase 4 · Model-RL 模型做正式评估
  基准对比：Phase 1 · Sim-RL（p1_ppo_v5，mean_reward=176,588）
  验收标准：mean_reward ≥ 176,588
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from backend.rl.fusion_env import FusionEnv

MBRL_BEST  = "data/rl_models/mbrl/mbrl_agent_best.zip"
MBRL_FINAL = "data/rl_models/mbrl/mbrl_agent_final.zip"
PPO_V5     = "data/rl_models/ppo_fusion_v5/final.zip"
N_EPISODES = 100

print("=" * 60)
print("Phase 4 · Model-RL — 100-episode FusionEnv 正式评估")
print("=" * 60)
print("\n【指标说明】")
print("  mean_reward  — 平均每局总奖励（Phase 1 · Sim-RL p1_ppo_v5 基准：176,588）")
print("  lawson_rate  — 劳森条件达成率（Lawson Criterion: n·T·τ > 1e27，等离子体点火标准）")
print("  disrupt_rate — 等离子体破裂率（越低越好，0% = 完全稳定）")
print("  ep_len       — 平均每局步数（满分 500 步）\n")

def evaluate(model_path: str, tag: str, n_episodes=N_EPISODES):
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"[{tag}] 加载失败：{e}")
        return None

    env = FusionEnv(max_steps=500)
    rewards, disruptions, lawsons, ep_lens = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        disrupted = False
        lawson_ok = False
        step_cnt = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_rew += rew
            step_cnt += 1
            if info.get("disrupted"):
                disrupted = True
            if info.get("lawson_achieved"):
                lawson_ok = True

        rewards.append(ep_rew)
        disruptions.append(disrupted)
        lawsons.append(lawson_ok)
        ep_lens.append(step_cnt)

        if (ep + 1) % 20 == 0:
            print(f"  [{tag}] {ep+1:3d}/{n_episodes} | "
                  f"mean_rew={np.mean(rewards[-20:]):.1f} | "
                  f"lawson={int(np.sum(lawsons[-20:]))}/20 | "
                  f"disrupt={int(np.sum(disruptions[-20:]))}/20")

    result = {
        "mean_reward":  np.mean(rewards),
        "std_reward":   np.std(rewards),
        "lawson_rate":  np.mean(lawsons) * 100,
        "disrupt_rate": np.mean(disruptions) * 100,
        "ep_len":       np.mean(ep_lens),
    }

    print(f"\n[{tag}] 评估完成（{n_episodes} episodes）：")
    print(f"  mean_reward  : {result['mean_reward']:.1f} ± {result['std_reward']:.1f}")
    print(f"  lawson_rate  : {result['lawson_rate']:.1f}%")
    print(f"  disrupt_rate : {result['disrupt_rate']:.1f}%")
    print(f"  ep_len       : {result['ep_len']:.1f}")

    threshold = 176588
    status = "✅ PASS" if result['mean_reward'] >= threshold else "❌ FAIL"
    print(f"\n  Phase 4 验收（≥ {threshold}）: {status}")
    return result

print("\n=== 评估 mbrl_agent_best.zip ===")
r_best = evaluate(MBRL_BEST, "MBRL_Best")

print("\n=== 评估 mbrl_agent_final.zip ===")
r_final = evaluate(MBRL_FINAL, "MBRL_Final")

print("\n=== 对比参考：PPO v5 基准 ===")
r_v5 = evaluate(PPO_V5, "PPO_v5_Base")

print("\n" + "=" * 60)
print("Phase 4 评估汇总")
print("=" * 60)
for name, r in [("MBRL Best", r_best), ("MBRL Final", r_final), ("PPO v5", r_v5)]:
    if r:
        print(f"  {name:<12} | reward={r['mean_reward']:>10.1f} | "
              f"lawson={r['lawson_rate']:5.1f}% | "
              f"disrupt={r['disrupt_rate']:5.1f}%")
print()
print("【名词备注】")
print("  劳森条件 — 核聚变点火最低要求（nτE 乘积超过临界值）")
print("  破裂 — 等离子体不稳定并突然熄灭的物理事件")
print("  MBRL — Model-Based RL，先学世界模型再训练策略")
print("  Dyna — MBRL 变体，在世界模型中虚拟rollout加速样本效率")
