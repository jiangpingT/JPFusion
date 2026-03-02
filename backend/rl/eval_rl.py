"""
eval_rl.py — FusionRL 评估脚本（Phase 1）

评估已训练 PPO 模型的性能：
  - 运行 N 个 episode，收集状态轨迹
  - 输出 Lawson 参数曲线、破裂统计
  - Gaming 检测（q95 < 2.1 比例）
  - 结果保存为 JSON（供 API 返回给前端）

运行命令：
    python -m backend.rl.eval_rl --model_path data/rl_models/ppo_fusion_xxx/final.zip
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from backend.rl.fusion_env import FusionEnv
from backend.rl.dynamics import denormalize_state, compute_tau_E
from backend.rl.rewards import compute_lawson_parameter, LAWSON_TARGET

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
EVAL_DIR   = DATA_DIR / "rl_eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# 最新评估结果缓存（供 API 直接返回）
_latest_trajectory: dict = {}


def get_latest_trajectory() -> dict:
    return _latest_trajectory


def evaluate_model(
    model_path: str,
    n_episodes: int = 10,
    max_steps: int  = 500,
    deterministic: bool = True,
) -> dict:
    """
    评估已训练模型。

    Returns:
        {
          "episodes": [ {trajectory, stats} ],  # 每个 episode 的轨迹
          "summary":  { mean_reward, disruption_rate, lawson_rate, gaming_proxy }
        }
    """
    global _latest_trajectory

    model = PPO.load(model_path)
    env   = FusionEnv(max_steps=max_steps)

    all_episodes = []
    rewards, lengths, disruptions, lawsons = [], [], [], []
    q95_low_steps = 0
    total_steps   = 0

    for ep_idx in range(n_episodes):
        obs, _ = env.reset()
        done   = False

        trajectory = {
            "steps":          [],
            "n_e":            [],
            "T_e":            [],
            "B":              [],
            "q95":            [],
            "beta_N":         [],
            "Ip":             [],
            "P_heat":         [],
            "lawson":         [],
            "tau_E":          [],
            "rewards":        [],
            "actions":        [],
        }

        ep_reward = 0.0
        step_count = 0
        disrupted  = False
        lawson_ok  = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            s_dict = denormalize_state(obs)
            tau_E  = compute_tau_E(s_dict["n_e"], s_dict["B"],
                                   s_dict["P_heat"], s_dict["Ip"])
            lawson = compute_lawson_parameter(s_dict["n_e"], s_dict["T_e"], tau_E)

            trajectory["steps"].append(step_count)
            trajectory["n_e"].append(s_dict["n_e"])
            trajectory["T_e"].append(s_dict["T_e"])
            trajectory["B"].append(s_dict["B"])
            trajectory["q95"].append(s_dict["q95"])
            trajectory["beta_N"].append(s_dict["beta_N"])
            trajectory["Ip"].append(s_dict["Ip"])
            trajectory["P_heat"].append(s_dict["P_heat"])
            trajectory["lawson"].append(lawson)
            trajectory["tau_E"].append(float(tau_E))
            trajectory["rewards"].append(float(rew))
            trajectory["actions"].append(action.tolist())

            ep_reward  += rew
            step_count += 1
            total_steps += 1

            if info.get("q95", 99) < 2.1:
                q95_low_steps += 1
            if info.get("disrupted"):
                disrupted = True
            if info.get("lawson_achieved"):
                lawson_ok = True

        rewards.append(ep_reward)
        lengths.append(step_count)
        disruptions.append(1 if disrupted else 0)
        lawsons.append(1 if lawson_ok else 0)

        ep_stats = {
            "episode":       ep_idx,
            "total_reward":  float(ep_reward),
            "n_steps":       step_count,
            "disrupted":     disrupted,
            "lawson_achieved": lawson_ok,
            "final_lawson":  trajectory["lawson"][-1] if trajectory["lawson"] else 0.0,
            "mean_q95":      float(np.mean(trajectory["q95"])) if trajectory["q95"] else 0.0,
        }
        all_episodes.append({"trajectory": trajectory, "stats": ep_stats})

    gaming_proxy = q95_low_steps / max(total_steps, 1)

    summary = {
        "n_episodes":       n_episodes,
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "mean_ep_length":   float(np.mean(lengths)),
        "disruption_rate":  float(np.mean(disruptions)),
        "lawson_rate":      float(np.mean(lawsons)),
        "gaming_proxy":     gaming_proxy,
        "lawson_target":    LAWSON_TARGET,
        "model_path":       model_path,
    }

    result = {
        "episodes": all_episodes,
        "summary":  summary,
    }

    # 缓存最新结果（API 查询用）
    _latest_trajectory = result

    # 保存到文件
    out_path = str(EVAL_DIR / "latest_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, default=float)

    # 打印摘要
    print(f"\n{'='*60}")
    print("【FusionRL 评估摘要】")
    print(f"  模型路径：{model_path}")
    print(f"  评估轮数：{n_episodes}")
    print(f"  平均奖励：{summary['mean_reward']:.1f} ± {summary['std_reward']:.1f}")
    print(f"  平均步数：{summary['mean_ep_length']:.0f}")
    print(f"  破裂率：  {summary['disruption_rate']*100:.1f}%")
    print(f"  劳森达成率（Lawson Criterion）：{summary['lawson_rate']*100:.1f}%")
    print(f"  Gaming 代理（q95<2.1 比例）：{gaming_proxy:.3f}")
    print(f"{'='*60}")

    print("\n【名词备注】")
    print("  劳森准则 (Lawson Criterion) — what: 核聚变点火条件 n*T*tau > 3e21"
          " / why: 用来判断等离子体是否具有科学价值 / how: 超过阈值才算真正意义上的聚变")
    print("  Gaming 代理 — what: q95 < 2.1 的步数比例 / why: 检测 agent 是否靠"
          "卡破裂边界刷奖励 / how: >0.5 表示 gaming 嫌疑，>0.8 几乎确定是 gaming")
    print("  破裂率 — what: episode 以破裂结束的比例 / why: 破裂=等离子体失控，"
          "是最严重的失败 / how: 越低越好，理想<10%")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionRL 评估")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps",  type=int, default=500)
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
    )
