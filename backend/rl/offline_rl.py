"""
offline_rl.py — Phase 3 · Offline-RL：历史数据离线强化学习（CQL）

【阶段定位】
  数据来源：Phase 1 · Sim-RL 生成的高质量轨迹 / EAST 历史放电数据
  学习方式：CQL（Conservative Q-Learning），d3rlpy 实现
  在四阶段流水线中的角色：
    ← 接收 Phase 1 · Sim-RL 的轨迹数据作为离线数据集
    → 输出最优策略，作为 Phase 4 · Model-RL 的初始 Actor

【核心挑战】
  1. Distribution shift — 覆盖惩罚防止 Q 值对数据集外动作过度乐观
  2. 奖励反推不准确 — τ_E 需从 P_heat / dT/dt 近似（非直接可观测）
  3. 类别不平衡 — 破裂事件占比极低（<3%），需特殊处理

运行命令：
    python -m backend.rl.offline_train
"""

import os
import time
import numpy as np
from pathlib import Path

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
CQL_DIR    = DATA_DIR / "rl_models" / "p3_cql"
CQL_DIR.mkdir(parents=True, exist_ok=True)


def train_cql(
    east_data_path: str,
    n_steps: int            = 100_000,
    conservative_weight: float = 5.0,
    batch_size: int         = 256,
    actor_lr: float         = 3e-4,
    critic_lr: float        = 3e-4,
    state_dict: dict        = None,
) -> str:
    """
    用 d3rlpy CQL 训练离线 RL 策略。

    Parameters:
        east_data_path      — EAST 数据文件路径
        n_steps             — 训练步数
        conservative_weight — CQL 覆盖惩罚系数（越大越保守）
        state_dict          — 外部状态字典（供 API 查询进度）

    Returns: 模型保存路径
    """
    try:
        import d3rlpy
    except ImportError:
        raise ImportError("请安装 d3rlpy: pip install d3rlpy")

    if state_dict is not None:
        state_dict["status"] = "loading_data"

    # ─── 加载数据 ─────────────────────────────────────────────────────────
    from backend.data.east_loader import load_itpa_iddb, load_synthetic_east_data
    from backend.data.replay_buffer import build_replay_buffer, to_d3rlpy_dataset

    print(f"\n[CQL] 加载 EAST 数据：{east_data_path}")
    try:
        df = load_itpa_iddb(east_data_path)
    except Exception as e:
        print(f"[CQL] 真实数据加载失败（{e}），使用合成数据")
        df = load_synthetic_east_data(n_shots=100)
    # 兼容 load_itpa_iddb 返回空 DataFrame 的情况
    if len(df) == 0:
        print("[CQL] 加载数据为空，回退到合成数据")
        df = load_synthetic_east_data(n_shots=100)

    buffer  = build_replay_buffer(df)
    dataset = to_d3rlpy_dataset(buffer)

    print(f"[CQL] 数据集：{buffer['n_transitions']} transitions，"
          f"{buffer['n_episodes']} episodes")

    if state_dict is not None:
        state_dict["status"] = "training"

    # ─── CQL 模型 ─────────────────────────────────────────────────────────
    # d3rlpy 2.x API
    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        alpha_learning_rate=1e-4,
        batch_size=batch_size,
        conservative_weight=conservative_weight,
        n_critics=2,
        actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=[128, 128]),
        critic_encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=[128, 128]),
    ).create()

    # ─── 训练进度回调 ────────────────────────────────────────────────────
    class CQLProgressCallback(d3rlpy.logging.FileAdapterFactory if hasattr(d3rlpy.logging, 'FileAdapterFactory') else object):
        pass

    def _progress_callback(algo, epoch, total_step):
        """每个 epoch 结束时的回调"""
        if state_dict is not None:
            state_dict["steps"]  = total_step
            state_dict["epoch"]  = epoch

    save_path = str(CQL_DIR / "p3_cql_fusion.d3")

    print(f"\n[CQL] 开始离线训练 | n_steps={n_steps} | "
          f"conservative_weight={conservative_weight}")
    print(f"[CQL] 保存路径：{save_path}\n")

    print("【指标说明】")
    print("  td_error    — 时序差分误差（Temporal Difference Error），越小说明 Q 函数越稳定")
    print("  conservative_loss — CQL 覆盖惩罚损失，防止 Q 值在 OOD 动作上过高估计")
    print("  actor_loss  — Actor 策略损失（最大化 Q 值）\n")

    cql.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=min(1000, n_steps // 10),
        save_interval=10,
        evaluators={},
        logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
    )

    cql.save(save_path)

    if state_dict is not None:
        state_dict["status"]     = "done"
        state_dict["model_path"] = save_path

    print(f"\n[CQL] 训练完成！模型已保存：{save_path}")

    # 在 FusionEnv 上简单评估
    _evaluate_cql_on_env(cql, n_episodes=5)

    return save_path


def _evaluate_cql_on_env(cql_model, n_episodes: int = 5):
    """在 FusionEnv 上评估 CQL 策略"""
    from backend.rl.fusion_env import FusionEnv
    from backend.rl.rewards import LAWSON_TARGET
    from backend.rl.dynamics import denormalize_state, compute_tau_E
    from backend.rl.rewards import compute_lawson_parameter

    env = FusionEnv(max_steps=500)
    rewards, lawsons, disruptions = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_rew = 0.0
        lawson_ok = False
        disrupted = False

        while not done:
            action = cql_model.predict(obs.reshape(1, -1))[0]
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_rew += rew
            if info.get("disrupted"):
                disrupted = True
            if info.get("lawson_achieved"):
                lawson_ok = True

        rewards.append(ep_rew)
        lawsons.append(lawson_ok)
        disruptions.append(disrupted)

    print(f"\n[CQL] FusionEnv 评估（{n_episodes} episodes）：")
    print(f"  平均奖励：{np.mean(rewards):.1f}")
    print(f"  破裂率：  {np.mean(disruptions)*100:.0f}%")
    print(f"  劳森达成：{np.mean(lawsons)*100:.0f}%")
