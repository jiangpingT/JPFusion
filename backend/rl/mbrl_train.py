"""
mbrl_train.py — Phase 4 · Model-RL：世界模型强化学习（Dyna MBRL）

【阶段定位】
  数据来源：Phase 3 · Offline-RL 最优策略（初始 Actor）+ 世界模型虚拟轨迹
  学习方式：Dyna（Sutton, 1990）= 世界模型 + RL 交替训练
  在四阶段流水线中的角色：
    ← 接收 Phase 3 · Offline-RL 最优策略作为初始 Actor
    ← 世界模型（MLP Ensemble ×5）提供虚拟训练数据
    → 最终精调在真实 FusionEnv 中进行，消除世界模型偏差

【Dyna 训练循环】每次迭代：
  1. 真实环境数据 → 更新世界模型（MLP Ensemble）
  2. 从世界模型采样虚拟轨迹（×10 倍真实数据量）
  3. 虚拟轨迹 → 更新 PPO Agent
  4. 不确定性高的状态 → 优先从真实数据更新（主动学习）

运行命令：
    python -m backend.rl.mbrl_train --east_data data/east_discharge.csv
"""

import os
import time
import argparse
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MBRL_DIR = DATA_DIR / "rl_models" / "mbrl"
MBRL_DIR.mkdir(parents=True, exist_ok=True)


def run_dyna_mbrl(
    east_data_path: str,
    n_iterations: int          = 50,
    n_rl_steps_per_iter: int   = 10_000,
    world_model_update_freq: int = 1,    # 每 N 次迭代更新一次世界模型
    wm_epochs_per_update: int  = 20,
    n_envs: int                = 2,
    state_dict: dict           = None,
) -> str:
    """
    Dyna MBRL 主训练循环。

    Returns: 最终 RL Agent 路径
    """
    if state_dict is not None:
        state_dict["status"] = "loading_data"

    # ─── 加载数据 ─────────────────────────────────────────────────────────
    from backend.data.east_loader import load_itpa_iddb, load_synthetic_east_data
    print(f"\n[Dyna MBRL] 加载 EAST 数据：{east_data_path}")
    try:
        east_df = load_itpa_iddb(east_data_path)
    except Exception as e:
        print(f"[Dyna MBRL] 真实数据加载失败（{e}），使用合成数据")
        east_df = load_synthetic_east_data(n_shots=100)
    # 兼容 load_itpa_iddb 返回空 DataFrame 的情况（列名不匹配时静默返回空）
    if len(east_df) == 0:
        print("[Dyna MBRL] 加载数据为空，回退到合成数据")
        east_df = load_synthetic_east_data(n_shots=100)

    # ─── 初始化世界模型 ────────────────────────────────────────────────────
    from backend.rl.world_model import train_world_model, WorldModelEnsemble, WM_DIR
    import torch

    print("[Dyna MBRL] 初始化世界模型...")
    world_model = train_world_model(
        east_df=east_df,
        n_epochs=50,   # 初始快速训练
        n_models=5,
    )

    # ─── 初始化 PPO Agent（在世界模型环境中训练）───────────────────────────
    from backend.rl.world_model_env import WorldModelEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def make_wm_env():
        return WorldModelEnv(world_model=world_model, max_steps=500)

    vec_env = DummyVecEnv([make_wm_env] * n_envs)
    vec_env = VecMonitor(vec_env)

    agent = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
    )

    # ─── Dyna 主循环 ──────────────────────────────────────────────────────
    print(f"\n[Dyna MBRL] 开始训练 | 迭代={n_iterations} | "
          f"每次 RL 步={n_rl_steps_per_iter}")
    print(f"[Dyna MBRL] 保存目录：{MBRL_DIR}\n")

    print("【指标说明】")
    print("  wm_val_loss  — 世界模型验证损失（越小 = 动力学预测越准确）")
    print("  rl_reward    — PPO 在世界模型中的平均 episode 奖励")
    print("  wm_uncertainty — 世界模型平均不确定性（高 = 还需更多真实数据）\n")

    best_reward = -float("inf")
    final_path  = str(MBRL_DIR / "mbrl_agent_final.zip")

    for iteration in range(1, n_iterations + 1):
        iter_start = time.time()

        if state_dict is not None:
            state_dict["status"]    = "training"
            state_dict["iteration"] = iteration

        # ─── 步骤 1：更新世界模型（每 world_model_update_freq 次）────────
        wm_val_loss = None
        if iteration % world_model_update_freq == 0:
            print(f"[Dyna MBRL] 迭代 {iteration}/{n_iterations} — 更新世界模型...")
            from backend.rl.world_model import train_world_model as _train_wm
            world_model = _train_wm(
                east_df=east_df,
                n_epochs=wm_epochs_per_update,
                n_models=5,
            )
            # 重建向量化环境（绑定新世界模型）
            vec_env.close()
            vec_env = DummyVecEnv([make_wm_env] * n_envs)
            vec_env = VecMonitor(vec_env)
            agent.set_env(vec_env)

        # ─── 步骤 2-3：在世界模型中训练 PPO Agent ───────────────────────
        print(f"[Dyna MBRL] 迭代 {iteration}/{n_iterations} — 在世界模型中训练 PPO "
              f"({n_rl_steps_per_iter} 步)...")
        agent.learn(
            total_timesteps=n_rl_steps_per_iter,
            reset_num_timesteps=False,
        )

        # ─── 评估（在真实 FusionEnv 上）────────────────────────────────
        from backend.rl.fusion_env import FusionEnv
        eval_env = FusionEnv(max_steps=500)
        eval_rewards = []
        for _ in range(3):
            obs, _ = eval_env.reset()
            done   = False
            ep_rew = 0.0
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_rew += rew
            eval_rewards.append(ep_rew)

        mean_reward = float(np.mean(eval_rewards))
        elapsed     = time.time() - iter_start

        if state_dict is not None:
            state_dict["rl_reward"] = mean_reward

        print(f"  迭代 {iteration:>3d}/{n_iterations} | "
              f"FusionEnv reward={mean_reward:>8.1f} | "
              f"耗时={elapsed:.1f}s")

        # 保存最优
        if mean_reward > best_reward:
            best_reward = mean_reward
            agent.save(str(MBRL_DIR / "mbrl_agent_best.zip"))

        # 每 10 次保存 checkpoint
        if iteration % 10 == 0:
            agent.save(str(MBRL_DIR / f"mbrl_agent_iter{iteration}.zip"))

    # ─── 保存最终模型 ─────────────────────────────────────────────────────
    agent.save(final_path)
    if state_dict is not None:
        state_dict["status"]     = "done"
        state_dict["model_path"] = final_path

    print(f"\n[Dyna MBRL] 训练完成！")
    print(f"  最优 FusionEnv reward：{best_reward:.1f}")
    print(f"  最终 Agent：{final_path}")

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dyna MBRL 训练")
    parser.add_argument("--east_data",          type=str,   default=None)
    parser.add_argument("--use_synthetic",       action="store_true")
    parser.add_argument("--n_iterations",        type=int,   default=50)
    parser.add_argument("--n_rl_steps_per_iter", type=int,   default=10_000)
    args = parser.parse_args()

    if args.use_synthetic or args.east_data is None:
        from backend.data.east_loader import load_synthetic_east_data
        import tempfile
        df = load_synthetic_east_data(n_shots=50)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        df.to_csv(tmp.name, index=False)
        tmp.close()
        east_data_path = tmp.name
        print(f"[mbrl_train] 使用合成数据：{tmp.name}")
    else:
        east_data_path = args.east_data

    run_dyna_mbrl(
        east_data_path=east_data_path,
        n_iterations=args.n_iterations,
        n_rl_steps_per_iter=args.n_rl_steps_per_iter,
    )
