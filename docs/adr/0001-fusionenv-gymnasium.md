# ADR-0001: 使用 Gymnasium 构建 FusionEnv 而非自定义仿真器

**状态**：接受

**日期**：2026-02

**决策人**：姜哥 / 阿策

---

## 背景

FusionRL Phase 1 需要一个托卡马克等离子体控制环境，供 RL Agent 在其中训练。有两条路：
1. 用现有托卡马克仿真软件（TRANSP、JINTRAC、PROCESS 等）
2. 自己用 ODE 动力学建一个简化仿真器

自定义仿真器还需要选择接口标准——是和 stable-baselines3 深度耦合，还是实现通用 RL 接口。

另外，本机是 Apple M4 Pro，没有 NVIDIA GPU，排除了需要 CUDA 的 Isaac Gym / Isaac Lab。

## 决策

用 **Gymnasium（OpenAI Gym 继任者）标准接口** 包装自定义 ODE 动力学模型，构建 `FusionEnv`。

具体实现：
- `FusionEnv(gymnasium.Env)` 继承 Gymnasium 标准接口（`reset` / `step` / `observation_space` / `action_space`）
- 动力学：ITER98pY2 τ_E 经验公式 + 1st-order 弛豫 ODE（不依赖外部仿真软件）
- 控制周期 dt = 0.01s，最大步数 500 步（模拟 5 秒真实放电时间）

## 后果

### 正面影响

- stable-baselines3 / d3rlpy 直接兼容，无需适配层
- Phase 4 WorldModelEnv 复用同一接口，代码一致性高
- 合成数据 + 真实数据均可无缝切换（dynamics.py 可注入校准系数）
- 无外部仿真软件依赖，可离线运行

### 负面影响 / 权衡

- ODE 简化导致物理精度有限（真实托卡马克动力学更复杂）
- 缺少 MHD 模式（撕裂模、气球模）的精确模拟，破裂条件只能用解析代理
- gymnasium 1.0.0 与 d3rlpy 有版本依赖冲突（需降级 gymnasium 1.2.3 → 1.0.0）

## 替代方案

### 方案 A：对接 TRANSP / JINTRAC 专业仿真软件

**放弃原因**：需要 Linux + 授权许可，安装配置极其复杂，M4 Mac 本机不可用。

### 方案 B：完全自定义接口（不用 Gymnasium）

**放弃原因**：失去 stable-baselines3 / d3rlpy 的直接兼容性，需要大量适配工作。

### 方案 C：用 PlasmaPy 物理引擎做动力学

**放弃原因**：PlasmaPy 专注于等离子体物理量计算（Debye 长度、等离子体频率等），
不提供时序动力学演化，需要自己写 ODE 无论如何。

---

*相关 ADR：[ADR-0002](0002-ppo-stable-baselines3.md)*
