# ADR-0006: 选择 Dyna 框架实现 MBRL 训练循环

**状态**：接受

**日期**：2026-02

**决策人**：姜哥 / 阿策

---

## 背景

Phase 4 需要设计 MBRL（Model-Based Reinforcement Learning）训练策略。
核心问题：**真实数据和虚拟数据（世界模型生成）怎么配合？**

可选框架：
1. **Dyna（Sutton, 1990）**：真实数据 + 世界模型交替，经典方法
2. **Dreamer V3**：端到端世界模型，在隐空间内想象训练
3. **PETS（Probabilistic Ensemble Trajectory Sampling）**：Ensemble 采样 + MPC 规划
4. **MBPO（Model-Based Policy Optimization）**：短 rollout + 混合 buffer

FusionEnv 的特点：
- 状态空间 7 维（低维，不需要图像表征）
- 动力学相对平滑（ODE 驱动，非离散跳变）
- 数据量有限（ITPA IDDB EAST 子集，1170+ 次放电）

## 决策

采用 **Dyna 框架**，实现"世界模型更新 + PPO 虚拟训练"的交替循环。

```
Dyna 循环（共 N 次迭代）：
  ┌──────────────────────────────────────────────────────┐
  │  每次迭代：                                           │
  │    1. 真实 EAST 数据 → 更新 World Model（20 epoch）   │
  │    2. 从 WorldModelEnv 采样虚拟轨迹 → PPO（10K 步）   │
  │    3. 在真实 FusionEnv 上评估（3 episode）            │
  │    4. 保存最优 checkpoint                             │
  └──────────────────────────────────────────────────────┘
```

关键设计决策：
- 虚拟轨迹 / 真实数据比例：10:1（每更新一次世界模型，做 10K 虚拟步）
- 不确定性惩罚：`UNCERTAINTY_PENALTY = -5.0`（防止 model exploitation）
- 虚拟 rollout 加噪声：`next_state = mean + 0.1 × std × noise`（防止确定性循环）
- 评估在真实 FusionEnv 上进行（世界模型是训练工具，不是评估标准）

## 后果

### 正面影响

- Dyna 实现简单，无需隐空间学习（适合 7D 低维状态）
- 与 PPO（stable-baselines3）直接兼容，WorldModelEnv 实现 Gymnasium 接口即可
- 真实数据校准世界模型，虚拟数据提升 PPO 样本效率，逻辑清晰
- M4 Pro 上世界模型推理 + PPO 更新均在 CPU/MPS 可接受范围

### 负面影响 / 权衡

- Dyna 的世界模型误差会累积（compound error）：长 rollout 虚拟轨迹误差大
- 当前实现用 WorldModelEnv 作为 SB3 VecEnv，每次世界模型更新需重建 VecEnv（有 overhead）
- 合成数据测试时世界模型 val_loss 高（数据太少），真实 EAST 数据才能验证

### 已知风险：Model Exploitation

Agent 可能学会在世界模型不准确的区域刷高 Q 值（即利用世界模型的漏洞）。
缓解措施：
1. 不确定性惩罚（已实现）
2. 定期用真实环境评估（Dyna 每轮都在真实 FusionEnv 测试）
3. 世界模型持续更新（每 1 次迭代更新一次，不让 Agent 过拟合老世界模型）

## 替代方案

### 方案 A：Dreamer V3

**放弃原因**：Dreamer 需要在隐空间（latent space）学习，适合图像输入。
FusionEnv 只有 7 维状态，不需要表征学习，Dreamer 的开销不值得。

### 方案 B：MBPO（Model-Based Policy Optimization）

**放弃原因**：MBPO 的核心创新是短 rollout（1-5 步）混入 replay buffer，
避免累积误差。实现需要精确控制 rollout 长度和 buffer 混合比例，
工程复杂度高，且对 FusionEnv 500 步 episode 的优势不明显。

### 方案 C：MPC + PETS（轨迹采样 + 模型预测控制）

**放弃原因**：MPC 每一步都需要多步前向规划（计算密集），
在 500 步 episode 的 FusionEnv 上每步规划开销太高。
适合短 horizon 控制任务，不适合长序列等离子体控制。

---

*相关 ADR：[ADR-0005](0005-mlp-ensemble-world-model.md) · [ADR-0002](0002-ppo-stable-baselines3.md)*
