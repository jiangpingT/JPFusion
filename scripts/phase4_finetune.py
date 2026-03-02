"""
phase4_finetune.py — Phase 4 · Model-RL v3 精调

【阶段定位】
  在真实 FusionEnv 上对 Phase 4 MBRL 最优检查点精调，消除 World Model Bias。
  ← 起点：p4_mbrl/p4_mbrl_v3_best.zip
  → 输出：p4_mbrl/p4_mbrl_v3_finetuned.zip
  → 副产品：data/trajectories/p4_finetune_buffer.npz（供 Phase 3 v3 回流）

【新增功能（pipeline v2）】
  精调期间收集真实 FusionEnv 轨迹，保存为 npz，供 Stage 5（Phase 3 v3）使用。
  这实现了"Phase 4 → Phase 3"的轨迹回流数据飞轮。

运行：
    venv/bin/python scripts/phase4_finetune.py [选项]

选项：
    --mbrl_best   输入模型路径（默认 p4_mbrl/p4_mbrl_v3_best.zip）
    --save_dir    输出目录（默认 data/rl_models/p4_mbrl）
    --traj_save   轨迹保存路径（默认 data/trajectories/p4_finetune_buffer.npz）
"""

import sys
import argparse
import json
import numpy as np
import shutil
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from backend.rl.fusion_env import FusionEnv

# ─── 默认路径 ─────────────────────────────────────────────────────────────────
DEFAULT_MBRL_BEST = "data/rl_models/p4_mbrl/p4_mbrl_v3_best.zip"
DEFAULT_SAVE_DIR  = Path("data/rl_models/p4_mbrl")
DEFAULT_TRAJ_SAVE = "data/trajectories/p4_finetune_buffer.npz"
N_FINETUNE_STEPS  = 300_000


class TrajectoryCollector(BaseCallback):
    """
    回调：在精调期间收集 (obs, act, rew, next_obs, terminal) 轨迹。
    精调结束后调用 save() 将轨迹保存为 npz（供 Phase 3 v3 数据回流）。
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.obs_buf      = []
        self.act_buf      = []
        self.rew_buf      = []
        self.next_obs_buf = []
        self.term_buf     = []
        self._prev_obs    = None

    def _on_step(self) -> bool:
        # clipped_actions = 实际传给环境的动作（已 clip 到 action_space 范围）
        # actions         = PPO 高斯分布原始采样（未 clip，不能用于 CQL 训练）
        clipped_actions = self.locals.get("clipped_actions")
        rewards         = self.locals.get("rewards")
        new_obs         = self.locals.get("new_obs")
        dones           = self.locals.get("dones")

        # obs_tensor 是 torch.Tensor，需要转 numpy
        obs_tensor = self.locals.get("obs_tensor")
        if obs_tensor is None or clipped_actions is None:
            return True

        import torch
        obs_np   = obs_tensor.cpu().numpy().reshape(-1, 7)
        act_np   = np.array(clipped_actions).reshape(-1, 3)
        rew_np   = np.array(rewards).reshape(-1)
        nobs_np  = np.array(new_obs).reshape(-1, 7)
        dones_np = np.array(dones).reshape(-1)

        for i in range(len(obs_np)):
            self.obs_buf.append(obs_np[i])
            self.act_buf.append(act_np[i])
            self.rew_buf.append(float(rew_np[i]))
            self.next_obs_buf.append(nobs_np[i])
            self.term_buf.append(float(dones_np[i]))

        return True

    def save(self, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            observations=np.array(self.obs_buf,      dtype=np.float32),
            actions=np.array(self.act_buf,           dtype=np.float32),
            rewards=np.array(self.rew_buf,           dtype=np.float32),
            next_observations=np.array(self.next_obs_buf, dtype=np.float32),
            terminals=np.array(self.term_buf,        dtype=np.float32),
            timeouts=np.zeros(len(self.term_buf),    dtype=np.float32),  # 精调轨迹无 timeout 区分
        )
        n = len(self.obs_buf)
        print(f"  精调轨迹已保存：{save_path}（{n} transitions）")
        return n


def main(mbrl_best: str = DEFAULT_MBRL_BEST,
         save_dir: str  = str(DEFAULT_SAVE_DIR),
         traj_save: str = DEFAULT_TRAJ_SAVE):

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 4 · Model-RL v3 精调 — FusionEnv 微调 + 轨迹回流收集")
    print("=" * 60)
    print("\n【指标说明】")
    print("  mean_reward  — 平均每局总奖励（目标 ≥ 176,588，Phase 1 v5 基准）")
    print("  精调前后对比：消除 World Model Bias，在真实仿真器中验证")
    print("  精调轨迹     — 收集真实 FusionEnv 数据，供 Phase 3 v3 数据回流\n")

    if not Path(mbrl_best).exists():
        raise FileNotFoundError(f"MBRL 最优模型不存在：{mbrl_best}\n"
                                f"请先运行 scripts/mbrl_expert_train.py")

    vec_env = make_vec_env(lambda: FusionEnv(max_steps=500), n_envs=4)
    vec_env = VecMonitor(vec_env)

    agent = PPO.load(mbrl_best, env=vec_env,
                     learning_rate=5e-5, clip_range=0.05, ent_coef=0.001)

    print(f"  从 {mbrl_best} 热启动")
    print(f"  精调步数：{N_FINETUNE_STEPS:,} | n_envs=4 | lr=5e-5")
    print(f"  轨迹回流保存：{traj_save}\n")

    # ─── 精调前评估（20 episodes）────────────────────────────────────────────
    eval_env = FusionEnv(max_steps=500)
    pre_rewards = []
    for _ in range(20):
        obs, _ = eval_env.reset()
        done = False; ep_rew = 0.0
        while not done:
            a, _ = agent.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = eval_env.step(a)
            done = terminated or truncated
            ep_rew += rew
        pre_rewards.append(ep_rew)
    print(f"精调前 20-episode: mean={np.mean(pre_rewards):.1f} ± {np.std(pre_rewards):.1f}")

    # ─── 精调（带轨迹收集 callback）─────────────────────────────────────────
    traj_collector = TrajectoryCollector()
    agent.learn(total_timesteps=N_FINETUNE_STEPS,
                reset_num_timesteps=False,
                callback=traj_collector)

    # ─── 保存精调模型 ────────────────────────────────────────────────────────
    # 备份已有 finetuned，防止结果更差时历史最优丢失
    _existing = save_dir_path / "p4_mbrl_v3_finetuned.zip"
    if _existing.exists():
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _bak = save_dir_path / f"backup_{_ts}_p4_mbrl_v3_finetuned.zip"
        shutil.copy2(_existing, _bak)
        print(f"  📦 备份已有 finetuned：{_bak.name}")

    finetuned_path = str(save_dir_path / "p4_mbrl_v3_finetuned.zip")
    agent.save(finetuned_path)
    print(f"\n精调完成，保存至：{finetuned_path}")

    # ─── 保存精调轨迹（供 Phase 3 v3 数据回流）──────────────────────────────
    n_traj = traj_collector.save(traj_save)
    print(f"  轨迹回流数据：{n_traj} transitions")

    # ─── 精调后 100-episode 正式评估 ─────────────────────────────────────────
    print("\n=== 100-episode 正式评估 ===")
    final_rewards, final_lawsons, final_disrupts = [], [], []
    for ep in range(100):
        obs, _ = eval_env.reset()
        done = False; ep_rew = 0.0; ep_law = False; ep_dis = False
        while not done:
            a, _ = agent.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = eval_env.step(a)
            done = terminated or truncated
            ep_rew += rew
            if info.get("lawson_achieved"): ep_law = True
            if info.get("disrupted"):       ep_dis = True
        final_rewards.append(ep_rew)
        final_lawsons.append(ep_law)
        final_disrupts.append(ep_dis)
        if (ep + 1) % 25 == 0:
            print(f"  {ep+1:3d}/100 | mean={np.mean(final_rewards[-25:]):.1f} | "
                  f"lawson={int(np.sum(final_lawsons[-25:]))}/25 | "
                  f"disrupt={int(np.sum(final_disrupts[-25:]))}/25")

    mean_rew   = float(np.mean(final_rewards))
    std_rew    = float(np.std(final_rewards))
    lawson_rt  = float(np.mean(final_lawsons))
    disrupt_rt = float(np.mean(final_disrupts))

    print(f"\n[Phase 4 v3 精调] 最终评估（100 episodes）：")
    print(f"  mean_reward  : {mean_rew:.1f} ± {std_rew:.1f}")
    print(f"  lawson_rate  : {lawson_rt*100:.1f}%")
    print(f"  disrupt_rate : {disrupt_rt*100:.1f}%")
    print(f"  精调前       : {np.mean(pre_rewards):.1f}")
    print(f"  提升         : {mean_rew - np.mean(pre_rewards):+.1f}")

    threshold = 176_588
    status = "✅ PASS" if mean_rew >= threshold else "❌ FAIL"
    print(f"\n  Phase 4 验收（≥ {threshold:,}）: {status}")

    # ─── 带分数命名存档（永久保留，可随时回滚）──────────────────────────────
    _ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    _reward_i = int(mean_rew)
    _lawson_i = int(lawson_rt * 100)
    archive_name = f"archive_{_ts}_reward{_reward_i}_lawson{_lawson_i}.zip"
    archive_path = save_dir_path / archive_name
    shutil.copy2(finetuned_path, archive_path)
    print(f"\n  📁 带分数存档：{archive_name}")

    # ─── 冠军挑战者机制 ─────────────────────────────────────────────────────
    champion_file = save_dir_path / "champion.json"
    champion_reward = -float("inf")
    if champion_file.exists():
        with open(champion_file) as f:
            champ = json.load(f)
        champion_reward = champ.get("mean_reward", -float("inf"))
        print(f"\n  👑 当前冠军：reward={champion_reward:.0f}（{champ.get('archive')}）")

    if mean_rew > champion_reward * 1.005:   # 至少提升 0.5% 才更新
        champ_data = {
            "mean_reward":  mean_rew,
            "std_reward":   std_rew,
            "lawson_rate":  lawson_rt,
            "disrupt_rate": disrupt_rt,
            "archive":      archive_name,
            "model_path":   str(archive_path),
            "timestamp":    _ts,
        }
        with open(champion_file, "w") as f:
            json.dump(champ_data, f, indent=2)
        print(f"  🏆 新冠军！reward={mean_rew:.0f} > 旧冠军 {champion_reward:.0f}（+{mean_rew-champion_reward:.0f}）")
        print(f"     冠军模型：{archive_name}")
    else:
        print(f"  🛡️  旧冠军守擂：{champion_reward:.0f} vs 新挑战者 {mean_rew:.0f}（差距不足 0.5%，不更新）")
        print(f"     旧冠军模型保留：{champ.get('archive') if champion_file.exists() else '无'}")

    return finetuned_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 v3 精调 + 轨迹回流收集")
    parser.add_argument("--mbrl_best",  type=str, default=DEFAULT_MBRL_BEST,
                        help="MBRL 最优模型路径（.zip）")
    parser.add_argument("--save_dir",   type=str, default=str(DEFAULT_SAVE_DIR),
                        help="精调模型输出目录")
    parser.add_argument("--traj_save",  type=str, default=DEFAULT_TRAJ_SAVE,
                        help="精调轨迹保存路径（.npz，供 Phase 3 v3 回流）")
    args = parser.parse_args()

    main(
        mbrl_best=args.mbrl_best,
        save_dir=args.save_dir,
        traj_save=args.traj_save,
    )
