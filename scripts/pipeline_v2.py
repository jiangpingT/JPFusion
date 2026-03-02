"""
pipeline_v2.py — 四阶段串行训练流水线 v2

【完整数据流】
  p2_sft_actor.pt
      ↓ warmstart_path
  [Stage 1] Phase 1 · Sim-RL (PPO v7)
      → data/rl_models/p1_ppo_v7/final.zip
      ↓ 收集 150 episodes 轨迹
  [Stage 2] Phase 3 · Offline-RL (CQL v2)
      → data/rl_models/p3_cql/p3_cql_v2.d3
      ↓ BC 蒸馏 150 episodes
  [Stage 3] data/rl_models/p3_cql/p3_to_p4_actor.pt
      ↓ 显式 key 映射热启动 PPO actor
  [Stage 4a] Phase 4 · Model-RL (MBRL v3)
      → data/rl_models/p4_mbrl/p4_mbrl_v3_best.zip
  [Stage 4b] Phase 4 精调
      → data/rl_models/p4_mbrl/p4_mbrl_v3_finetuned.zip
      + data/trajectories/p4_finetune_buffer.npz
      ↓ 轨迹回流
  [Stage 5] Phase 3 · Offline-RL (CQL v3)
      → data/rl_models/p3_cql/p3_cql_v3.d3

【多次迭代】
  --n_iterations N：Stage 2-5 循环 N 次（默认 1）
  每次迭代 Phase 3/4 版本号递增（v2→v3→v4...）
  实现"数据飞轮"——Phase 4 真实轨迹持续回流 Phase 3，质量螺旋上升。

运行：
    cd /Users/mlamp/Workspace/fusion-platform
    source venv/bin/activate
    python scripts/pipeline_v2.py [--n_iterations 1] [--skip_stage1]

选项：
    --n_iterations   Stage 2-5 循环次数（默认 1）
    --skip_stage1    跳过 Phase 1 训练（已有 p1_ppo_v7 时使用）
    --skip_sft       跳过 SFT 热启动警告（直接用随机初始化 Phase 1）
    --p1_steps       Phase 1 训练步数（默认 300,000）
    --cql_n_episodes CQL 专家 episode 数（默认 150）
    --cql_n_steps    CQL 训练步数（默认 200,000）
    --distill_ep     蒸馏 episode 数（默认 150）
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

# ─── 路径常量 ──────────────────────────────────────────────────────────────────
P2_SFT_ACTOR  = "data/rl_models/p2_sft/p2_sft_actor.pt"
P1_V7_PATH    = "data/rl_models/p1_ppo_v7/final.zip"
P3_CQL_DIR    = "data/rl_models/p3_cql"
P4_MBRL_DIR   = "data/rl_models/p4_mbrl"
TRAJ_DIR      = "data/trajectories"

PYTHON = sys.executable  # 使用当前 venv 的 Python，保证环境一致


def banner(text: str):
    print("\n" + "═" * 64)
    print(f"  {text}")
    print("═" * 64)


def run_stage(name: str, cmd: list, env_check: str = None) -> float:
    """
    运行一个流水线 Stage，返回耗时（秒）。
    失败时打印错误并 raise SystemExit。
    """
    banner(f"▶  {name}")
    print(f"  命令：{' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ Stage 失败：{name}")
        print(f"   退出码：{result.returncode}")
        print(f"   耗时：{elapsed:.1f}s")
        raise SystemExit(f"流水线中止于：{name}")

    # 验证输出文件是否存在
    if env_check and not Path(env_check).exists():
        print(f"\n⚠️ Stage 完成但输出文件未找到：{env_check}")
        print("   流水线继续，但后续 Stage 可能失败...")

    banner(f"✅ {name} 完成（耗时 {elapsed/60:.1f} min）")
    return elapsed


def main(n_iterations: int, skip_stage1: bool, skip_sft: bool,
         p1_steps: int, cql_n_episodes: int, cql_n_steps: int, distill_ep: int):

    total_start = time.time()
    stage_times = {}

    print("=" * 64)
    print("  FusionRL 四阶段串行训练流水线 v2")
    print("=" * 64)
    print(f"\n  n_iterations : {n_iterations}")
    print(f"  skip_stage1  : {skip_stage1}")
    print(f"  Python       : {PYTHON}")
    print(f"\n【数据流】p2_sft_actor → Phase1 → Phase3 → 蒸馏 → Phase4 → Phase3 v3")
    print(f"【迭代】Stage 2-5 循环 {n_iterations} 次（数据飞轮）\n")

    # ────────────────────────────────────────────────────────────────────────
    # Stage 0：检查 SFT Actor
    # ────────────────────────────────────────────────────────────────────────
    banner("Stage 0 — 检查 Phase 2 SFT Actor")
    sft_available = Path(P2_SFT_ACTOR).exists()
    if sft_available:
        print(f"  ✅ SFT Actor 存在：{P2_SFT_ACTOR}")
    else:
        if skip_sft:
            print(f"  ⚠️ SFT Actor 不存在（已设 --skip_sft，继续用随机初始化）")
        else:
            print(f"  ⚠️ SFT Actor 不存在：{P2_SFT_ACTOR}")
            print(f"  Phase 1 将从随机初始化开始（SFT 热启动跳过）")
            print(f"  如需 SFT 热启动，请先运行：")
            print(f"    python -m backend.rl.bc_pretrain --east_data data/east_discharge.csv")
        sft_available = False

    # ────────────────────────────────────────────────────────────────────────
    # Stage 1：Phase 1 · Sim-RL（PPO v7）
    # ────────────────────────────────────────────────────────────────────────
    if skip_stage1:
        banner("Stage 1 — Phase 1 · Sim-RL（跳过，使用已有 v7）")
        if not Path(P1_V7_PATH).exists():
            print(f"  ❌ --skip_stage1 但 {P1_V7_PATH} 不存在！")
            raise SystemExit("请先训练 Phase 1 v7 或移除 --skip_stage1 标志")
        print(f"  ✅ 使用已有模型：{P1_V7_PATH}")
    else:
        cmd_p1 = [
            PYTHON, "-m", "backend.rl.train_rl",
            "--n_steps",  str(p1_steps),
            "--version",  "v7",
            "--n_envs",   "4",
            "--lr",       "3e-4",
            "--ent_coef", "0.005",
        ]
        if sft_available:
            cmd_p1 += ["--warmstart", P2_SFT_ACTOR]

        t = run_stage("Stage 1 — Phase 1 · Sim-RL (PPO v7)", cmd_p1, P1_V7_PATH)
        stage_times["Stage 1"] = t

    # ────────────────────────────────────────────────────────────────────────
    # Stage 2-5 迭代（数据飞轮）
    # ────────────────────────────────────────────────────────────────────────
    for iteration in range(1, n_iterations + 1):
        iter_tag = f"（迭代 {iteration}/{n_iterations}）" if n_iterations > 1 else ""

        # CQL 版本命名：第 1 次 v2，第 2 次 v3，...
        cql_version = iteration + 1    # v2, v3, v4...
        cql_name    = f"p3_cql_v{cql_version}.d3"
        cql_path    = f"{P3_CQL_DIR}/{cql_name}"
        cql_actor   = f"{P3_CQL_DIR}/p3_to_p4_actor.pt"
        traj_path   = f"{TRAJ_DIR}/p4_finetune_buffer_iter{cql_version}.npz"

        # 首次迭代无 extra_buffer；后续迭代用上次 Phase 4 轨迹
        prev_traj = f"{TRAJ_DIR}/p4_finetune_buffer_iter{cql_version - 1}.npz"

        # ── Stage 2：Phase 3 · CQL ──────────────────────────────────────────
        cmd_cql = [
            PYTHON, "scripts/cql_expert_train.py",
            "--source_model", P1_V7_PATH,
            "--save_dir",     P3_CQL_DIR,
            "--save_name",    cql_name,
            "--n_episodes",   str(cql_n_episodes),
            "--n_steps",      str(cql_n_steps),
        ]
        if iteration > 1 and Path(prev_traj).exists():
            cmd_cql += ["--extra_buffer", prev_traj]
            print(f"  [迭代 {iteration}] 加入上次 Phase 4 轨迹回流：{prev_traj}")

        t = run_stage(f"Stage 2 — Phase 3 · CQL {cql_name} {iter_tag}", cmd_cql, cql_path)
        stage_times[f"Stage 2 iter{iteration}"] = t

        # ── Stage 3：Phase 3→4 BC 蒸馏 ──────────────────────────────────────
        cmd_distill = [
            PYTHON, "scripts/p3_to_p4_distill.py",
            "--cql_path",     cql_path,
            "--save_path",    cql_actor,
            "--n_distill_ep", str(distill_ep),
        ]
        t = run_stage(f"Stage 3 — Phase 3→4 BC 蒸馏 {iter_tag}", cmd_distill, cql_actor)
        stage_times[f"Stage 3 iter{iteration}"] = t

        # ── Stage 4a：Phase 4 MBRL ────────────────────────────────────────
        # 第 2 轮起：WM 增量更新（--finetune_wm），合并旧 Phase 1 + 新 Phase 4 轨迹
        cmd_mbrl = [
            PYTHON, "scripts/mbrl_expert_train.py",
            "--p3_actor",     cql_actor,
            "--source_model", P1_V7_PATH,
        ]
        if iteration > 1 and Path(prev_traj).exists():
            cmd_mbrl += ["--finetune_wm", "--extra_episodes_npz", prev_traj]
            print(f"  [WM 增量更新] 合并 Phase 4 轨迹：{prev_traj}")
        mbrl_best = f"{P4_MBRL_DIR}/p4_mbrl_v3_best.zip"
        t = run_stage(f"Stage 4a — Phase 4 · MBRL v3 {iter_tag}", cmd_mbrl, mbrl_best)
        stage_times[f"Stage 4a iter{iteration}"] = t

        # ── Stage 4b：Phase 4 精调 + 轨迹回流收集 ───────────────────────────
        cmd_ft = [
            PYTHON, "scripts/phase4_finetune.py",
            "--mbrl_best",  mbrl_best,
            "--save_dir",   P4_MBRL_DIR,
            "--traj_save",  traj_path,
        ]
        finetuned_path = f"{P4_MBRL_DIR}/p4_mbrl_v3_finetuned.zip"
        t = run_stage(f"Stage 4b — Phase 4 精调 + 轨迹回流 {iter_tag}", cmd_ft, finetuned_path)
        stage_times[f"Stage 4b iter{iteration}"] = t

        # ── Stage 5：Phase 3 v3（加入 Phase 4 轨迹重训）────────────────────
        if Path(traj_path).exists():
            next_cql_version = cql_version + 1
            next_cql_name    = f"p3_cql_v{next_cql_version}.d3"
            next_cql_path    = f"{P3_CQL_DIR}/{next_cql_name}"

            cmd_cql_v3 = [
                PYTHON, "scripts/cql_expert_train.py",
                "--source_model", P1_V7_PATH,
                "--save_dir",     P3_CQL_DIR,
                "--save_name",    next_cql_name,
                "--extra_buffer", traj_path,
                "--n_episodes",   str(cql_n_episodes),
                "--n_steps",      str(cql_n_steps),
            ]
            t = run_stage(f"Stage 5 — Phase 3 {next_cql_name}（+Phase 4 回流）{iter_tag}",
                          cmd_cql_v3, next_cql_path)
            stage_times[f"Stage 5 iter{iteration}"] = t
        else:
            banner(f"Stage 5 — 跳过（Phase 4 轨迹文件未找到：{traj_path}）")

    # ────────────────────────────────────────────────────────────────────────
    # 汇总报告
    # ────────────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    banner("🎉 流水线 v2 全部完成！")

    print("\n【耗时统计】")
    for stage_name, t in stage_times.items():
        print(f"  {stage_name:<30s} : {t/60:>6.1f} min")
    print(f"  {'总计':<30s} : {total_elapsed/60:>6.1f} min")

    print("\n【产物路径】")
    print(f"  Phase 1 v7  : {P1_V7_PATH}")
    for i in range(1, n_iterations + 1):
        v = i + 1
        print(f"  Phase 3 v{v}  : {P3_CQL_DIR}/p3_cql_v{v}.d3")
        print(f"  Phase 4 v3  : {P4_MBRL_DIR}/p4_mbrl_v3_finetuned.zip")
        if Path(f"{P3_CQL_DIR}/p3_cql_v{v+1}.d3").exists():
            print(f"  Phase 3 v{v+1} : {P3_CQL_DIR}/p3_cql_v{v+1}.d3（含 Phase 4 回流）")

    print("\n【验收建议】")
    print("  Phase 1 v7  mean_reward ≥ 170,000")
    print("  Phase 3 v2  mean_reward ≥ 190,000")
    print("  Phase 4 v3  mean_reward ≥ 182,000")
    print("  Phase 3 v3  mean_reward ≥ 195,000")

    print("\n【名词备注】")
    print("  数据飞轮 — Phase 4 真实轨迹回流 Phase 3，每次迭代数据质量螺旋提升")
    print("  BC 蒸馏  — 行为克隆，将 CQL 策略转换为 SB3 PPO 兼容格式")
    print("  串行流水线 — 各阶段依序执行，前一阶段输出作为后一阶段输入")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionRL 四阶段串行训练流水线 v2")
    parser.add_argument("--n_iterations",   type=int, default=1,
                        help="Stage 2-5 数据飞轮循环次数（默认 1）")
    parser.add_argument("--skip_stage1",    action="store_true",
                        help="跳过 Phase 1 训练（已有 p1_ppo_v7 时使用）")
    parser.add_argument("--skip_sft",       action="store_true",
                        help="跳过 SFT 热启动（直接随机初始化 Phase 1）")
    parser.add_argument("--p1_steps",       type=int, default=1_000_000,
                        help="Phase 1 训练步数（默认 1,000,000）")
    parser.add_argument("--cql_n_episodes", type=int, default=150,
                        help="CQL 专家 episode 数（默认 150）")
    parser.add_argument("--cql_n_steps",    type=int, default=200_000,
                        help="CQL 训练步数（默认 200,000）")
    parser.add_argument("--distill_ep",     type=int, default=150,
                        help="BC 蒸馏 episode 数（默认 150）")
    args = parser.parse_args()

    main(
        n_iterations=args.n_iterations,
        skip_stage1=args.skip_stage1,
        skip_sft=args.skip_sft,
        p1_steps=args.p1_steps,
        cql_n_episodes=args.cql_n_episodes,
        cql_n_steps=args.cql_n_steps,
        distill_ep=args.distill_ep,
    )
