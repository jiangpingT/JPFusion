"""
train_rl.py — Phase 1 · Sim-RL：仿真环境在线强化学习（PPO）

【阶段定位】
  数据来源：FusionEnv ODE 仿真器（实时在线交互，无需历史数据）
  学习方式：PPO（近端策略优化，on-policy 强化学习）
  在四阶段流水线中的角色：
    ← 可接收 Phase 2 · Sim-SFT 的 BC 预训练权重作为热启动
    → 生成高质量轨迹数据，供 Phase 3 · Offline-RL 使用

【算法】stable-baselines3 PPO，Lawson 准则（n·T·τ > 1e27）驱动奖励

运行命令：
    python -m backend.rl.train_rl --n_steps 200000

训练文件保存到：data/rl_models/p1_ppo_{version}/
"""

import os
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from backend.rl.fusion_env import FusionEnv

# ─── 模型保存路径 ─────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = DATA_DIR / "rl_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─── 训练状态（供 API 查询）────────────────────────────────────────────────────
_rl_training_state = {
    "status":         "idle",      # idle / training / done / error
    "episode":        0,
    "total_steps":    0,
    "mean_reward":    None,
    "mean_ep_length": None,
    "disruption_rate": None,
    "lawson_rate":    None,
    "history":        [],          # [{episode, mean_reward, ...}, ...]
    "model_path":     None,
    "error_msg":      None,
    "start_time":     None,
    "elapsed_s":      0,
}


def get_rl_training_state() -> dict:
    return dict(_rl_training_state)


class FusionRLCallback(BaseCallback):
    """
    自定义回调：
      - 收集 episode 统计并更新 _rl_training_state
      - 检测 gaming 行为（q95 < 2.1 比例）
      - 可选 WebSocket 推送（异步）
    """

    def __init__(self, eval_env: FusionEnv, ws_callback=None,
                 eval_freq: int = 5000, verbose: int = 1,
                 save_dir: str = None):
        super().__init__(verbose)
        self.eval_env    = eval_env
        self.ws_callback = ws_callback
        self.eval_freq   = eval_freq
        self._save_dir   = save_dir          # ★ 用于保存 best.zip
        self._best_mean_reward = -float("inf")
        self._episode_rewards = []
        self._episode_lengths = []
        self._disruption_count = 0
        self._lawson_count     = 0
        self._last_eval_step   = 0

    def _on_step(self) -> bool:
        # 收集每个 episode 结束时的统计
        for info in self.locals.get("infos", []):
            if info.get("episode"):
                ep = info["episode"]
                self._episode_rewards.append(ep["r"])
                self._episode_lengths.append(ep["l"])

        # 检查破裂 / Lawson 成功
        for info in self.locals.get("infos", []):
            if info.get("disrupted"):
                self._disruption_count += 1
            if info.get("lawson_achieved"):
                self._lawson_count += 1

        # 定频评估 + 状态推送
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            self._run_eval()

        return True

    def _run_eval(self):
        """运行 5 episode 评估，更新全局状态"""
        n_eval = 5
        rewards, lengths, disruptions, lawsons = [], [], [], []
        q95_low_count = 0
        total_steps   = 0

        for _ in range(n_eval):
            obs, _ = self.eval_env.reset()
            done   = False
            ep_rew = 0.0
            ep_len = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_rew += rew
                ep_len += 1
                total_steps += 1

                # Gaming 检测：q95 < 2.1 的步数比例
                if info.get("q95", 99) < 2.1:
                    q95_low_count += 1

            rewards.append(ep_rew)
            lengths.append(ep_len)
            disruptions.append(1 if info.get("disrupted") else 0)
            lawsons.append(1 if info.get("lawson_achieved") else 0)

        mean_reward    = float(np.mean(rewards))
        mean_length    = float(np.mean(lengths))
        disruption_rate = float(np.mean(disruptions))
        lawson_rate    = float(np.mean(lawsons))
        gaming_proxy   = q95_low_count / max(total_steps, 1)

        now = time.time()
        record = {
            "timestep":       self.num_timesteps,
            "mean_reward":    mean_reward,
            "mean_ep_length": mean_length,
            "disruption_rate": disruption_rate,
            "lawson_rate":    lawson_rate,
            "gaming_proxy":   gaming_proxy,  # > 0.5 = gaming 嫌疑
            "timestamp":      now,
        }

        _rl_training_state["episode"]        = len(_rl_training_state["history"])
        _rl_training_state["total_steps"]    = self.num_timesteps
        _rl_training_state["mean_reward"]    = mean_reward
        _rl_training_state["mean_ep_length"] = mean_length
        _rl_training_state["disruption_rate"] = disruption_rate
        _rl_training_state["lawson_rate"]    = lawson_rate
        _rl_training_state["elapsed_s"]      = now - (_rl_training_state["start_time"] or now)
        _rl_training_state["history"].append(record)

        # ★ Best checkpoint：每次评估若刷新最高 mean_reward，保存 best.zip
        if mean_reward > self._best_mean_reward and self._save_dir is not None:
            self._best_mean_reward = mean_reward
            best_path = str(Path(self._save_dir) / "best.zip")
            self.model.save(best_path)
            best_tag = f" 🏅 new best!"
        else:
            best_tag = ""

        if self.verbose >= 1:
            print(
                f"[FusionRL] step={self.num_timesteps:>8d} | "
                f"mean_rew={mean_reward:>8.1f} | "
                f"ep_len={mean_length:>5.0f} | "
                f"disrupt={disruption_rate:.2f} | "
                f"lawson={lawson_rate:.2f} | "
                f"gaming={gaming_proxy:.3f}{best_tag}"
            )

        # 指标说明（首次打印）
        if len(_rl_training_state["history"]) == 1:
            print("\n【指标说明】")
            print("  mean_rew    — 每次评估 5 个 episode 的平均总奖励（越高越好）")
            print("  ep_len      — 平均每 episode 步数（越长 = 越少破裂）")
            print("  disrupt     — 破裂率（0.0 最优，>0.5 说明控制失败）")
            print("  lawson      — 劳森准则（Lawson Criterion）达成率（>0 = 有科学价值）")
            print("  gaming      — gaming 代理（q95 < 2.1 步数比例，>0.5 = 嫌疑）\n")

        # WebSocket 推送（可选）
        if self.ws_callback:
            import asyncio
            try:
                asyncio.get_event_loop().run_until_complete(
                    self.ws_callback({"type": "rl_progress", **record})
                )
            except Exception:
                pass


def train_rl(
    n_steps: int         = 200_000,
    n_envs: int          = 4,
    learning_rate: float = 3e-4,
    ent_coef: float      = 0.005,
    gamma: float         = 0.99,
    max_ep_steps: int    = 500,
    warmstart_path: str  = None,   # BC 预训练权重路径（Phase 2C）
    ppo_warmstart: str   = None,   # 已有 PPO 模型路径（热启动继续训练）
    ws_callback=None,
    version: str         = None,
) -> str:
    """
    启动 PPO 训练。

    Returns: 保存的模型路径
    """
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = MODELS_DIR / f"p1_ppo_{version}"
    save_dir.mkdir(parents=True, exist_ok=True)

    _rl_training_state["status"]     = "training"
    _rl_training_state["start_time"] = time.time()
    _rl_training_state["history"]    = []
    _rl_training_state["error_msg"]  = None

    try:
        # ─── 创建向量化训练环境 ───────────────────────────────────────────
        def make_env():
            return FusionEnv(max_steps=max_ep_steps)

        vec_env = make_vec_env(make_env, n_envs=n_envs)
        vec_env = VecMonitor(vec_env)

        # ─── 评估环境（单个，确定性策略）────────────────────────────────
        eval_env = FusionEnv(max_steps=max_ep_steps)

        # ─── 创建 PPO 模型（或从已有 PPO 热启动）──────────────────────────
        if ppo_warmstart and os.path.exists(ppo_warmstart):
            print(f"[FusionRL] 从已有 PPO 模型热启动: {ppo_warmstart}")
            model = PPO.load(ppo_warmstart, env=vec_env)
            # 可选更新超参数
            model.ent_coef = ent_coef
            model.learning_rate = learning_rate
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=gamma,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=ent_coef,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[128, 128]),
                verbose=0,
            )

        # ─── 热启动（来自 BC 预训练）────────────────────────────────────
        if warmstart_path and os.path.exists(warmstart_path):
            print(f"[FusionRL] 从 BC 预训练权重热启动: {warmstart_path}")
            import torch
            bc_weights   = torch.load(warmstart_path, map_location="cpu")
            policy_state = model.policy.state_dict()
            # BCActorMLP key → PPO policy key 显式映射
            # （key 名不同但形状相同：net.X.* → mlp_extractor.policy_net.X.*）
            key_map = {
                "net.0.weight": "mlp_extractor.policy_net.0.weight",
                "net.0.bias":   "mlp_extractor.policy_net.0.bias",
                "net.2.weight": "mlp_extractor.policy_net.2.weight",
                "net.2.bias":   "mlp_extractor.policy_net.2.bias",
                "net.4.weight": "action_net.weight",
                "net.4.bias":   "action_net.bias",
            }
            mapped = 0
            for bc_key, ppo_key in key_map.items():
                if bc_key in bc_weights and ppo_key in policy_state:
                    bc_w, pp_w = bc_weights[bc_key], policy_state[ppo_key]
                    if bc_w.shape == pp_w.shape:
                        policy_state[ppo_key] = bc_w
                        mapped += 1
            model.policy.load_state_dict(policy_state, strict=False)
            print(f"[FusionRL] BC 热启动完成（映射 {mapped}/6 个参数层）")

        # ─── 回调 ────────────────────────────────────────────────────────
        fusion_callback = FusionRLCallback(
            eval_env=eval_env,
            ws_callback=ws_callback,
            eval_freq=5000,
            verbose=1,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=20000,
            save_path=str(save_dir),
            name_prefix="p1_ppo",
        )

        print(f"\n[FusionRL] 开始训练 | n_steps={n_steps} | n_envs={n_envs} | "
              f"lr={learning_rate} | ent_coef={ent_coef}")
        print(f"[FusionRL] 模型保存目录: {save_dir}\n")

        # ─── 训练 ────────────────────────────────────────────────────────
        model.learn(
            total_timesteps=n_steps,
            callback=[fusion_callback, checkpoint_callback],
            reset_num_timesteps=True,
        )

        # ─── 保存最终模型 ─────────────────────────────────────────────────
        final_path = str(save_dir / "final.zip")
        model.save(final_path)
        _rl_training_state["status"]     = "done"
        _rl_training_state["model_path"] = final_path

        print(f"\n[FusionRL] 训练完成！最终模型: {final_path}")

        # 保存训练历史
        history_path = str(save_dir / "history.json")
        with open(history_path, "w") as f:
            json.dump(_rl_training_state["history"], f, indent=2)

        return final_path

    except Exception as e:
        _rl_training_state["status"]    = "error"
        _rl_training_state["error_msg"] = str(e)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionRL PPO 训练")
    parser.add_argument("--n_steps",  type=int,   default=200_000, help="总训练步数")
    parser.add_argument("--n_envs",   type=int,   default=4,       help="并行环境数")
    parser.add_argument("--lr",       type=float, default=3e-4,    help="学习率")
    parser.add_argument("--ent_coef", type=float, default=0.005,   help="熵系数")
    parser.add_argument("--version",  type=str,   default=None,    help="版本标签")
    parser.add_argument("--warmstart", type=str,  default=None,    help="BC 预训练权重路径")
    parser.add_argument("--ppo_warmstart", type=str, default=None, help="已有 PPO 模型路径（热启动）")
    args = parser.parse_args()

    train_rl(
        n_steps=args.n_steps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        version=args.version,
        warmstart_path=args.warmstart,
        ppo_warmstart=args.ppo_warmstart,
    )
