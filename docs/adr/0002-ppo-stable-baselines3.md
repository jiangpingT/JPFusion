# ADR-0002: 选择 PPO + stable-baselines3 作为在线 RL 方案

**状态**：接受

**日期**：2026-02

**决策人**：姜哥 / 阿策

---

## 背景

FusionRL Phase 1 需要选择在线 RL 算法和训练框架。需要权衡：
- 样本效率（每个 episode 多少样本能学到有效策略）
- 连续动作空间支持（3D 动作向量）
- M4 Pro CPU 训练性能（无 CUDA）
- 与 Gymnasium 接口的兼容性
- 超参数调节难度（姜哥和阿策都更熟悉 PPO）

JPRobot 后空翻同样使用 PPO（stable-baselines3），经验可直接复用。

## 决策

使用 **PPO（近端策略优化）+ stable-baselines3** 作为 Phase 1 在线 RL 方案。

具体配置：
```python
PPO(
    "MlpPolicy",
    policy_kwargs=dict(net_arch=[128, 128]),
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.005,
    clip_range=0.2,
)
```

并行环境数：4（M4 Pro 10 性能核，留给系统 2 核，4 envs × 2 进程 = 8 核占用）

## 后果

### 正面影响

- PPO 对超参数不敏感，调试容易
- stable-baselines3 自带 VecEnv / Monitor / Callback 生态
- 与 JPRobot 经验完全一致，gaming 检测逻辑可直接迁移
- 支持热启动（Phase 2C SFT 权重 → PPO Actor 初始化）
- MPS 加速通过 PyTorch 自动启用（SB3 2.7+ 支持）

### 负面影响 / 权衡

- PPO 是 on-policy 算法，样本效率低于 SAC（off-policy）
- 4 个并行环境与 JPRobot 的 12 个相比偏少（等离子体 ODE 比 PyBullet 快，可以适当增加）
- ent_coef 过大会导致策略随机、gaming 不固化；过小会导致探索不足

## 替代方案

### 方案 A：SAC（软 Actor-Critic，off-policy）

**放弃原因**：SAC 样本效率更高，但连续动作空间需要更复杂的调参，
M4 Pro 上 replay buffer 的内存开销也更大。未来可以作为 Phase 1 的改进版本。

### 方案 B：TD3（双延迟深度确定性策略梯度）

**放弃原因**：TD3 对 FusionEnv 这类稀疏奖励环境（Lawson 一次性大奖励）不友好，
探索性不如 PPO + entropy bonus。

### 方案 C：自定义 RL 训练循环（不用 SB3）

**放弃原因**：工程量过大，且失去 SB3 的调试工具和稳定性保证。
项目目标是科学深度，不是重新实现 RL 框架。

---

*相关 ADR：[ADR-0001](0001-fusionenv-gymnasium.md) · [ADR-0003](0003-lawson-reward-anti-gaming.md)*
