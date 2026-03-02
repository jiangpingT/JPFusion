# 架构决策记录（ADR）索引

> ADR（Architecture Decision Record）—— 记录每一个重要架构决策的来龙去脉，
> 让未来的开发者（和阿策）理解"为什么这么设计"，而不只是"是什么"。
>
> 规范参考：[Michael Nygard ADR 格式](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)

---

## ADR 格式说明

每个 ADR 文件包含以下固定章节：

| 章节 | 内容 |
|------|------|
| **标题** | 一句话描述决策 |
| **状态** | 提议 / 接受 / 废弃 / 已取代 |
| **背景** | 为什么需要做这个决策 |
| **决策** | 我们选择了什么 |
| **后果** | 正面和负面影响 |
| **替代方案** | 考虑过但放弃的选项 |

---

## 决策记录列表

| 编号 | 标题 | 状态 | 日期 |
|------|------|------|------|
| [ADR-0001](0001-fusionenv-gymnasium.md) | 使用 Gymnasium 构建 FusionEnv 而非自定义仿真器 | 接受 | 2026-02 |
| [ADR-0002](0002-ppo-stable-baselines3.md) | 选择 PPO + stable-baselines3 作为在线 RL 方案 | 接受 | 2026-02 |
| [ADR-0003](0003-lawson-reward-anti-gaming.md) | Lawson 准则驱动奖励 + 防 gaming 设计 | 接受 | 2026-02 |
| [ADR-0004](0004-east-itpa-iddb.md) | 使用 ITPA IDDB 作为 EAST 真实数据来源 | 接受 | 2026-02 |
| [ADR-0005](0005-mlp-ensemble-world-model.md) | 使用 MLP Ensemble 实现世界模型 | 接受 | 2026-02 |
| [ADR-0006](0006-dyna-mbrl-training-loop.md) | 选择 Dyna 框架实现 MBRL 训练循环 | 接受 | 2026-02 |

---

## 如何新增 ADR

1. 复制模板：`cp docs/adr/TEMPLATE.md docs/adr/000N-title.md`
2. 填写各章节（背景最重要，必须诚实记录真实动机）
3. 更新本索引表格
4. Commit：`git commit -m "docs: add ADR-000N <title>"`

---

*文档维护：阿策 · FusionLab*
