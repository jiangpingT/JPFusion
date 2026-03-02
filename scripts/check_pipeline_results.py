"""
check_pipeline_results.py — pipeline v2 结果一键摘要

运行：venv/bin/python scripts/check_pipeline_results.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ─── 检查各阶段模型文件 ─────────────────────────────────────────────────────
MODELS = {
    "Phase 1 v7  (PPO)": "data/rl_models/p1_ppo_v7/final.zip",
    "Phase 3 v2  (CQL)": "data/rl_models/p3_cql/p3_cql_v2.d3",
    "Phase 3→4 蒸馏 actor": "data/rl_models/p3_cql/p3_to_p4_actor.pt",
    "Phase 4 v3  (MBRL best)": "data/rl_models/p4_mbrl/p4_mbrl_v3_best.zip",
    "Phase 4 v3  (精调)": "data/rl_models/p4_mbrl/p4_mbrl_v3_finetuned.zip",
    "Phase 4 精调轨迹": "data/trajectories/p4_finetune_buffer.npz",
    "Phase 3 v3  (CQL)": "data/rl_models/p3_cql/p3_cql_v3.d3",
}

print("=" * 60)
print("Pipeline v2 — 模型文件状态")
print("=" * 60)
for name, path in MODELS.items():
    p = Path(path)
    if p.exists():
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  ✅ {name:30s} {size_mb:.1f} MB")
    else:
        print(f"  ❌ {name:30s} 未生成")

# ─── 对已有模型做 100-episode 评估 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("100-episode 评估（deterministic policy）")
print("=" * 60)

from backend.rl.fusion_env import FusionEnv

def evaluate_model(model, n_ep=100):
    env = FusionEnv(max_steps=500)
    rewards, lawsons, disrupts = [], [], []
    for _ in range(n_ep):
        obs, _ = env.reset()
        done = False; ep_rew = 0.0; ep_law = False; ep_dis = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(a)
            done = term or trunc
            ep_rew += rew
            if info.get("lawson_achieved"): ep_law = True
            if info.get("disrupted"):       ep_dis = True
        rewards.append(ep_rew)
        lawsons.append(ep_law)
        disrupts.append(ep_dis)
    return np.mean(rewards), np.std(rewards), np.mean(lawsons)*100, np.mean(disrupts)*100

# Phase 1 v7
p1_path = Path("data/rl_models/p1_ppo_v7/final.zip")
if p1_path.exists():
    print("\n[Phase 1 v7 PPO]")
    from stable_baselines3 import PPO
    model = PPO.load(str(p1_path))
    m, s, l, d = evaluate_model(model)
    print(f"  mean_reward : {m:.0f} ± {s:.0f}")
    print(f"  lawson_rate : {l:.1f}%")
    print(f"  disrupt_rate: {d:.1f}%")
    status = "✅ PASS" if m >= 170000 else "❌ FAIL"
    print(f"  验收(≥170k) : {status}")

# Phase 3 v2 CQL
p3v2_path = Path("data/rl_models/p3_cql/p3_cql_v2.d3")
if p3v2_path.exists():
    print("\n[Phase 3 v2 CQL]")
    try:
        import d3rlpy
        cql = d3rlpy.load_learnable(str(p3v2_path))
        class CQLWrapper:
            def __init__(self, cql): self.cql = cql
            def predict(self, obs, deterministic=True):
                return self.cql.predict(obs.reshape(1,-1))[0], None
        m, s, l, d = evaluate_model(CQLWrapper(cql))
        print(f"  mean_reward : {m:.0f} ± {s:.0f}")
        print(f"  lawson_rate : {l:.1f}%")
        print(f"  disrupt_rate: {d:.1f}%")
        status = "✅ PASS" if m >= 190000 else "❌ FAIL"
        print(f"  验收(≥190k) : {status}")
    except Exception as e:
        print(f"  评估失败: {e}")

# Phase 4 v3 精调
p4_path = Path("data/rl_models/p4_mbrl/p4_mbrl_v3_finetuned.zip")
if p4_path.exists():
    print("\n[Phase 4 v3 MBRL 精调]")
    from stable_baselines3 import PPO
    model = PPO.load(str(p4_path))
    m, s, l, d = evaluate_model(model)
    print(f"  mean_reward : {m:.0f} ± {s:.0f}")
    print(f"  lawson_rate : {l:.1f}%")
    print(f"  disrupt_rate: {d:.1f}%")
    status = "✅ PASS" if m >= 182000 else "❌ FAIL"
    print(f"  验收(≥182k) : {status}")

# Phase 3 v3 CQL
p3v3_path = Path("data/rl_models/p3_cql/p3_cql_v3.d3")
if p3v3_path.exists():
    print("\n[Phase 3 v3 CQL（加入 Phase 4 轨迹）]")
    try:
        import d3rlpy
        cql = d3rlpy.load_learnable(str(p3v3_path))
        class CQLWrapper:
            def __init__(self, cql): self.cql = cql
            def predict(self, obs, deterministic=True):
                return self.cql.predict(obs.reshape(1,-1))[0], None
        m, s, l, d = evaluate_model(CQLWrapper(cql))
        print(f"  mean_reward : {m:.0f} ± {s:.0f}")
        print(f"  lawson_rate : {l:.1f}%")
        print(f"  disrupt_rate: {d:.1f}%")
        status = "✅ PASS" if m >= 195000 else "❌ FAIL"
        print(f"  验收(≥195k) : {status}")
    except Exception as e:
        print(f"  评估失败: {e}")

print("\n" + "=" * 60)
print("评估完成")
print("=" * 60)
