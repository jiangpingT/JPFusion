"""
strategy_validator.py — DTW 策略有效性验证（Phase 2B）

把 RL Agent 的控制轨迹，与 EAST 真实专家操作轨迹做 DTW 比较。
DTW 距离 < 10.0 → 策略有物理意义
DTW 距离 > 50.0 → gaming 嫌疑
"""

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from stable_baselines3 import PPO
from backend.rl.fusion_env import FusionEnv
from backend.rl.dynamics import denormalize_state


# DTW 比较的维度（控制量三轴）
_COMPARE_KEYS = ["Ip", "P_heat", "n_e"]
# 归一化范围（用于 DTW 比较前统一量纲）
_NORMALIZE_RANGES = {
    "Ip":     (0.5e6, 2.0e6),
    "P_heat": (0.5e6, 20e6),
    "n_e":    (1e19,  1e20),
}


def _normalize_traj(traj: list[dict], keys: list[str]) -> np.ndarray:
    """将轨迹 dict 列表转为归一化的 N×3 数组（用于 DTW）"""
    arr = []
    for step in traj:
        row = []
        for k in keys:
            lo, hi = _NORMALIZE_RANGES[k]
            val = step.get(k, lo)
            row.append((val - lo) / max(hi - lo, 1e-10))
        arr.append(row)
    return np.array(arr, dtype=np.float64)


def _extract_east_trajectory(df: pd.DataFrame, shot_id=None) -> list[dict]:
    """
    从 EAST DataFrame 中提取一条"良好放电"轨迹。

    良好放电：无破裂 + q95 > 2.0（整个 episode）。
    如果指定 shot_id，提取该 shot；否则随机选一条无破裂的。
    """
    if shot_id is not None:
        sub = df[df["shot_id"] == shot_id]
    else:
        # 选 Ip/n_e 都有效且 q95 健康的 shot
        shot_ids = df["shot_id"].unique()
        candidates = []
        for sid in shot_ids:
            sub = df[df["shot_id"] == sid]
            if (sub["n_e"].notna().all() and sub["Ip"].notna().all()
                    and len(sub) >= 20):
                if "q95" in sub.columns and sub["q95"].notna().any():
                    if sub["q95"].min() > 2.0:
                        candidates.append(sid)
                else:
                    candidates.append(sid)
        if not candidates:
            # fallback：用第一个 shot
            sub = df[df["shot_id"] == shot_ids[0]]
        else:
            sub = df[df["shot_id"] == candidates[0]]

    traj = []
    for _, row in sub.iterrows():
        traj.append({
            "n_e":    float(row.get("n_e", 0)),
            "Ip":     float(row.get("Ip", 0)),
            "P_heat": float(row.get("P_heat", 0)),
        })
    return traj


def _run_agent_trajectory(model_path: str, n_steps: int = 200) -> list[dict]:
    """运行 RL Agent 收集一条轨迹"""
    model = PPO.load(model_path)
    env   = FusionEnv(max_steps=n_steps)
    obs, _ = env.reset()
    done   = False

    traj = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        traj.append({
            "n_e":    info.get("n_e", 0),
            "Ip":     info.get("Ip", 0),
            "P_heat": info.get("P_heat", 0),
        })

    return traj


def compare_with_east(
    model_path: str,
    east_df: pd.DataFrame,
    n_episodes: int = 5,
) -> dict:
    """
    比较 RL Agent 轨迹 vs EAST 专家轨迹（DTW 距离）。

    Returns:
        {
            "dtw_score": float,         # 平均 DTW 距离（越小越好）
            "dtw_scores": [float],      # 每个 episode 的 DTW
            "judgment": str,            # "有物理意义" / "gaming 嫌疑" / "需要改进"
            "agent_traj_length": int,   # Agent 轨迹平均长度
            "east_traj_length": int,    # EAST 参考轨迹长度
        }
    """
    # 提取 EAST 参考轨迹
    east_traj = _extract_east_trajectory(east_df)
    east_arr  = _normalize_traj(east_traj, _COMPARE_KEYS)

    dtw_scores = []
    traj_lengths = []

    for _ in range(n_episodes):
        agent_traj = _run_agent_trajectory(model_path)
        traj_lengths.append(len(agent_traj))

        if len(agent_traj) < 5:
            dtw_scores.append(999.0)  # 轨迹太短，惩罚
            continue

        agent_arr = _normalize_traj(agent_traj, _COMPARE_KEYS)
        dist, _ = fastdtw(agent_arr, east_arr, dist=euclidean)
        dtw_scores.append(float(dist))

    mean_dtw = float(np.mean(dtw_scores))
    mean_len = float(np.mean(traj_lengths))

    # 判断
    if mean_dtw < 10.0:
        judgment = "有物理意义（DTW 距离与 EAST 专家接近）"
    elif mean_dtw < 50.0:
        judgment = "需要改进（控制轨迹与 EAST 存在偏差）"
    else:
        judgment = "gaming 嫌疑（控制轨迹与 EAST 专家差异极大）"

    print(f"\n[strategy_validator] DTW 平均距离：{mean_dtw:.2f}")
    print(f"  判断：{judgment}")
    print(f"  Agent 轨迹平均长度：{mean_len:.0f} 步")
    print(f"  EAST 参考轨迹长度：{len(east_traj)} 步")

    return {
        "dtw_score":          mean_dtw,
        "dtw_scores":         dtw_scores,
        "judgment":           judgment,
        "agent_traj_length":  mean_len,
        "east_traj_length":   len(east_traj),
        "east_shot_id":       int(east_df["shot_id"].iloc[0]) if "shot_id" in east_df else 0,
    }
