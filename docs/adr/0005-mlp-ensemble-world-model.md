# ADR-0005: 使用 MLP Ensemble 实现世界模型

**状态**：接受

**日期**：2026-02

**决策人**：姜哥 / 阿策

---

## 背景

Phase 4 Dyna MBRL 需要一个"世界模型"——用 EAST 数据学习等离子体动力学，
让 RL Agent 在世界模型里训练，而不是每次都依赖真实环境。

世界模型的核心需求：
1. 预测下一步状态（next_state = f(state, action)）
2. 同时预测即时奖励（reward = g(state, action)）
3. **提供不确定性估计**——不确定的区域告诉 Agent"这里我不确定，别走太远"
4. 计算效率高——M4 Pro CPU/MPS，不能用耗显存的架构

## 决策

使用 **5 个独立 MLP 的 Ensemble（集成）** 实现世界模型。

```python
class WorldModelEnsemble(nn.Module):
    def __init__(self, n_models=5):
        self.models = nn.ModuleList([MLP(10, 8) for _ in range(n_models)])

    def forward(self, state, action):
        x = cat(state, action)              # 输入：10 维
        preds = stack([m(x) for m in self.models])  # (5, B, 8)
        mean = preds.mean(dim=0)            # 均值预测
        var  = preds.var(dim=0)             # 不确定性
        return mean, var                    # 输出：next_state(7) + reward(1)
```

每个 MLP 独立训练（独立 optimizer），不共享参数，保证 ensemble 多样性。
不确定性用于：
1. `WorldModelEnv`：不确定性高 → 施加惩罚（`UNCERTAINTY_PENALTY = -5.0`）
2. Dyna 循环：不确定性热图告知哪些区域需要更多真实数据

隐藏层：`[256, 256]`，激活：SiLU（比 ReLU 更光滑，适合物理量预测）

## 后果

### 正面影响

- Ensemble 方差是不确定性的免费估计，无需额外训练（比 BNN/MC Dropout 简单）
- n_models=5 在 M4 Pro 上推理速度约 0.5ms/step（MPS 加速）
- 结构简单，调试友好
- 不确定性惩罚防止 Agent 利用世界模型漏洞（model exploitation）

### 负面影响 / 权衡

- 5 个 MLP 训练时间是 1 个的 5 倍（但每个 MLP 训练是独立的，可并行）
- Ensemble 方差低估尾部不确定性（OOD 样本的不确定性估计偏低）
- 需要足够多样的训练数据，5 个 MLP 才能分歧（合成数据过于均匀时 Ensemble 退化）
- 合成数据 5 epoch 测试：val_loss 较高（300+），这是数据少 + 训练轮次少导致，
  真实数据 + 100 epoch 后预计 < 5% 误差

## 替代方案

### 方案 A：概率神经网络（BNN / SWAG）

**放弃原因**：BNN 训练复杂，需要 variational inference；SWAG 需要特殊优化器。
两者在 M4 Pro 上的计算开销比 Ensemble 更大，且调试难度高。

### 方案 B：RSSM（Dreamer 的循环世界模型）

**放弃原因**：RSSM 需要序列处理，对等离子体控制的长序列（500 步）效果好，
但实现复杂度极高，且对 EAST 数据量（1170 次放电）可能 overfit。
适合作为未来 Phase 5 的研究方向。

### 方案 C：高斯过程（Gaussian Process）

**放弃原因**：GP 在高维输入（10 维）时计算复杂度 O(n³)，推理速度极慢，
不适合 Dyna 循环中的高频虚拟轨迹采样。

---

*相关 ADR：[ADR-0006](0006-dyna-mbrl-training-loop.md) · [ADR-0004](0004-east-itpa-iddb.md)*
