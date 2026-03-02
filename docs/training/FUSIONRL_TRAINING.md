# FusionRL 四阶段训练手册

> 从 demo 到真正的科学深度——完整操作指南
>
> 状态：Phase 1 已验证 · Phase 2-4 已实现待真实数据

---

## 前置准备

### 依赖安装

```bash
cd fusion-platform
source venv/bin/activate
pip install -r backend/requirements.txt
```

主要依赖：

| 包 | 用途 | 版本 |
|----|----|------|
| `gymnasium` | FusionEnv 环境框架 | 1.0.0 |
| `stable-baselines3` | PPO 在线强化学习 | 2.7.1 |
| `d3rlpy` | CQL 离线强化学习 | 2.8.1 |
| `scipy` | τ_E 系数拟合 | latest |
| `fastdtw` | DTW 轨迹距离计算 | 0.3.4 |
| `torch` | MLP Ensemble 世界模型 | 2.5.1 |

### 目录结构（运行时）

```
data/
├── rl_models/
│   ├── ppo_fusion_{version}/   # PPO 训练模型
│   │   ├── final.zip           # 最终权重
│   │   └── history.json        # 训练历史
│   ├── bc_sft/
│   │   └── bc_sft_actor.pt     # SFT 行为克隆权重
│   ├── cql_offline/
│   │   └── cql_fusion.d3       # CQL 模型
│   ├── world_model/
│   │   └── world_model.pt      # MLP Ensemble 权重
│   └── mbrl/
│       └── mbrl_agent_best.zip # Dyna MBRL 最优策略
└── rl_eval/
    └── latest_eval.json        # 最新评估结果
```

---

## Phase 1 — 纯 FusionEnv + PPO（推荐先跑）

**目标**：不依赖任何真实数据，靠 Lawson 准则奖励驱动，训练托卡马克等离子体基础控制策略。

完全类比 JPRobot 后空翻——靠奖励设计，而非数据。

### 环境说明

**状态空间（7 维，归一化 0~1）**

| 维度 | 变量 | 物理含义 | 典型范围 |
|------|------|----------|----------|
| 0 | `n_e_norm` | 电子密度 | 1×10¹⁹ ~ 1×10²⁰ m⁻³ |
| 1 | `T_e_norm` | 电子温度 | 1×10⁶ ~ 5×10⁷ K |
| 2 | `B_norm` | 磁场强度 | 2.0 ~ 6.0 T |
| 3 | `q95_norm` | 安全因子 q95 | 2.0 ~ 8.0 |
| 4 | `beta_N_norm` | 归一化 beta | 0.0 ~ 4.0 |
| 5 | `Ip_norm` | 等离子体电流 | 0.5 ~ 2.0 MA |
| 6 | `P_heat_norm` | 加热功率 | 0.5 ~ 20 MW |

**动作空间（3 维，[-0.1, 0.1]）**

| 维度 | 变量 | 物理含义 |
|------|------|----------|
| 0 | `delta_P_heat` | 加热功率增量（每步最多 ±1.95 MW） |
| 1 | `delta_n_fuel` | 燃料注入增量（控制密度） |
| 2 | `delta_Ip` | 电流增量（影响 q95 和 β_N） |

**4 个破裂终止条件（触发任意一个 → episode done）**

| 条件 | 物理机制 | 典型特征 |
|------|----------|----------|
| `q95 < 2.0` | Kruskal-Shafranov 不稳定 | 最常见破裂，约占 60% |
| `β_N > 3.5` | Troyon 稳定极限 | 压强过大触发气球模 |
| `n_e > n_Greenwald` | Greenwald 密度极限 | 高密度辐射骤冷 |
| 锁模代理 `> 5.0` | 锁定模不稳定性 | 低 q95 + 高 Ip 同时出现 |

**奖励设计（防 gaming）**

```
total_reward = Lawson 对数奖励（可负）
             + 效率惩罚（-0.05 × P_heat_MW，每步）
             + 稳定性惩罚（q95 < 2.5 或 β_N 裕度 < 0.5 时，负值）
             + 成功一次性奖励（首次跨越 Lawson 准则门槛：+500）
```

> **防 gaming 关键**：效率惩罚和稳定性惩罚都是负值，无正的 per-step 奖励流。
> Agent 不能靠"卡边界 + 不行动"刷分。

### 运行训练

**命令行方式（推荐调试）**

```bash
source venv/bin/activate

# 标准训练（200K 步，4 个并行环境）
python -m backend.rl.train_rl --n_steps 200000 --n_envs 4

# 自定义参数
python -m backend.rl.train_rl \
  --n_steps 500000 \
  --n_envs 4 \
  --lr 3e-4 \
  --ent_coef 0.005 \
  --version v1
```

**API 方式（前端触发）**

```bash
curl -X POST http://localhost:8000/api/rl/train \
  -H "Content-Type: application/json" \
  -d '{
    "n_steps": 200000,
    "n_envs": 4,
    "learning_rate": 0.0003,
    "ent_coef": 0.005,
    "max_ep_steps": 500
  }'
```

### 训练监控

实时指标（每 5000 步评估一次）：

| 指标 | 说明 | 目标值 |
|------|------|--------|
| `mean_rew` | 平均 episode 奖励 | 上升趋势 |
| `ep_len` | 平均步数 | > 400（越长越少破裂） |
| `disrupt` | 破裂率 | < 0.1（10%） |
| `lawson` | Lawson 准则达成率 | > 0（有科学价值） |
| `gaming` | q95 < 2.1 步数比例 | < 0.3（< 50% 则嫌疑） |

### 运行评估

```bash
python -m backend.rl.eval_rl \
  --model_path data/rl_models/ppo_fusion_v1/final.zip \
  --n_episodes 20
```

评估输出示例：

```
【FusionRL 评估摘要】
  平均奖励：  18742.3 ± 2341.1
  平均步数：  487
  破裂率：    4.0%
  劳森达成率：35.0%
  Gaming 代理（q95<2.1 比例）：0.012
```

### 验收标准

- [ ] `episode_reward` 呈上升趋势（至少 100K 步后）
- [ ] `disruption_rate < 0.1`（破裂率低于 10%）
- [ ] `gaming_proxy < 0.3`（无明显 gaming）
- [ ] `lawson_rate > 0`（至少有部分 episode 达到 Lawson 准则）

### 已知 Gaming 模式

| 模式 | 特征 | 解法 |
|------|------|------|
| 功率暴涨破裂 | P_heat 拉满 → T_e 飙升 → β_N 超限 | 效率惩罚（已实现）|
| q95 卡边 | q95 = 2.01 长期游荡 | 稳定性惩罚（已实现）|
| 持续收入流 | 每步正奖励 → agent 不想结束 episode | 无 per-step 正奖励（已实现）|

---

## Phase 2A — 物理参数校准（τ_E 系数拟合）

**目标**：用 EAST 真实放电数据，拟合 ITER98pY2 能量约束时间经验公式系数，让 FusionEnv 动力学更准确。

### 获取 ITPA IDDB 数据

ITPA IDDB（国际托卡马克物理活动数据库）包含 EAST 数据子集，1170+ 次放电，无需注册。

```bash
# 方法 1：从 Harvard Dataverse 下载
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OQTQXJ
# 下载后放到 data/east/ 目录

# 方法 2：使用合成数据（快速测试，无需真实数据）
python -c "
from backend.data.east_loader import load_synthetic_east_data
df = load_synthetic_east_data(n_shots=100, steps_per_shot=200)
df.to_csv('data/east_synthetic.csv', index=False)
print('合成数据已生成：data/east_synthetic.csv')
"
```

### 运行系数拟合

```bash
curl -X POST http://localhost:8000/api/calibration/fit-tau-e \
  -H "Content-Type: application/json" \
  -d '{"data_path": "data/east_synthetic.csv"}'
```

预期响应（合成数据）：

```json
{
  "success": true,
  "coefficients": {
    "C": 0.0562,
    "a": 0.410,
    "c": 0.150,
    "d": -0.690,
    "e": 0.930
  },
  "r_squared": 1.0,
  "n_samples": 19800,
  "message": "拟合成功（R²=1.0000）"
}
```

### 验收标准

- [ ] R² > 0.8（合成数据应 ≈ 1.0，真实 EAST 数据期望 > 0.7）

---

## Phase 2B — 策略有效性验证（DTW）

**目标**：把 RL Agent 控制轨迹与 EAST 真实专家轨迹做 DTW（动态时间规整）比较。

DTW 距离 < 10.0 → 策略有物理意义
DTW 距离 > 50.0 → gaming 嫌疑

```bash
curl -X POST http://localhost:8000/api/calibration/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "data/rl_models/ppo_fusion_v1/final.zip",
    "east_data_path": "data/east_synthetic.csv",
    "n_episodes": 5
  }'
```

### 验收标准

- [ ] DTW 分数 < 10.0（对比随机策略 > 50.0）

---

## Phase 2C — SFT 行为克隆预训练

**目标**：用 EAST 专家操作轨迹做 SFT（Supervised Fine-Tuning）监督学习，给 RL Agent 提供有物理意义的初始策略。

> **SFT = 行为克隆（Behavior Cloning, BC）**
> 类比 LLM 训练流程：先 SFT 让模型"说人话"，再 RLHF 对齐。
> 此处：先 SFT 让 Agent 学会基础控制，再 PPO 优化目标。

```bash
# 命令行
python -m backend.rl.bc_pretrain \
  --east_data data/east_synthetic.csv \
  --n_epochs 500 \
  --lr 1e-3

# API
curl -X POST http://localhost:8000/api/rl/bc-pretrain \
  -H "Content-Type: application/json" \
  -d '{
    "east_data_path": "data/east_synthetic.csv",
    "n_epochs": 500,
    "lr": 0.001
  }'
```

### 热启动 PPO

SFT 完成后，用 BC 权重热启动 PPO，避免冷启动长期探索：

```bash
python -m backend.rl.train_rl \
  --n_steps 200000 \
  --warmstart data/rl_models/bc_sft/bc_sft_actor.pt \
  --version v2_bc_warmstart
```

### 验收标准

- [ ] SFT val_loss < 5×10⁻⁴（合成数据）
- [ ] 热启动后，前 10K 步 mean_reward > 冷启动前 50K 步的值

---

## Phase 3 — Offline RL（CQL 保守 Q 学习）

**目标**：完全从 EAST 历史放电数据中学习策略，不需要 FusionEnv 仿真器。

> **CQL（Conservative Q-Learning）**：离线 RL 算法，在 Q 值外推上施加保守惩罚，
> 防止 distribution shift（在未见过的状态上 Q 值过高估计）。

### 核心挑战

| 挑战 | 说明 | 缓解措施 |
|------|------|----------|
| Distribution shift | EAST 专家区域 ≠ CQL 探索区域 | conservative_weight 调大（默认 5.0）|
| 奖励反推不准 | τ_E 不直接可观测 | 从 P_heat 和时序 T_e 近似计算 |
| 类别不平衡 | 破裂事件极少（<5%） | terminal 标记确保 done 信号准确 |

### 运行训练

```bash
# 使用合成数据（快速验证流程）
python -m backend.rl.offline_train --use_synthetic --n_steps 100000

# 使用真实 EAST 数据
python -m backend.rl.offline_train \
  --east_data data/east/iddb_east.csv \
  --n_steps 200000 \
  --conservative_weight 5.0
```

### 验收标准

- [ ] CQL 在 FusionEnv 上评估 mean_reward > Phase 1 PPO 的 50%

---

## Phase 4 — World Model + Dyna MBRL

**目标**：用 EAST 数据训练 MLP Ensemble 世界模型，让 Agent 在世界模型里训练（Dyna 循环），每次有新数据时世界模型更新，策略随之迭代。

### 世界模型架构

```
输入：(state[7], action[3]) = 10 维
输出：(next_state[7], reward[1]) = 8 维

5 个独立 MLP（各自独立训练）
均值 = 最优预测
方差 = 不确定性（高 → 优先从真实数据学习）
```

### Dyna 训练循环

```
每次迭代（共 50 次）：
  1. 真实 EAST 数据 → 更新世界模型（20 epoch）
  2. 世界模型采样虚拟轨迹 → 更新 PPO（10K 步）
  3. 在真实 FusionEnv 上评估（3 episode）
  4. 保存最优 checkpoint
```

### 运行训练

```bash
# 步骤 1：单独训练世界模型
python -c "
from backend.rl.world_model import train_world_model
from backend.data.east_loader import load_synthetic_east_data
df = load_synthetic_east_data(n_shots=200)
model = train_world_model(east_df=df, n_epochs=100, n_models=5)
"

# 步骤 2：Dyna MBRL 完整训练
python -m backend.rl.mbrl_train \
  --use_synthetic \
  --n_iterations 50 \
  --n_rl_steps_per_iter 10000

# 或使用真实数据
python -m backend.rl.mbrl_train \
  --east_data data/east/iddb_east.csv \
  --n_iterations 50
```

### 不确定性热图查询

```bash
curl http://localhost:8000/api/rl/world-model-uncertainty
```

### 验收标准

- [ ] 世界模型预测误差 < 5%（在测试集上）
- [ ] Dyna MBRL 在 FusionEnv 评估 reward ≥ Phase 1 PPO

---

## 完整训练流水线（推荐顺序）

```bash
# === Phase 1：建立 RL 基线 ===
python -m backend.rl.train_rl --n_steps 200000 --version v1
python -m backend.rl.eval_rl --model_path data/rl_models/ppo_fusion_v1/final.zip

# === Phase 2：数据校准（需 EAST 数据，或用合成数据测试流程） ===
# 2A 物理校准
python -c "
from backend.data.east_loader import load_synthetic_east_data
from backend.calibration.physics_calibrator import fit_tau_e_coefficients
df = load_synthetic_east_data(n_shots=100)
result = fit_tau_e_coefficients(df)
print(result)
"

# 2C SFT 行为克隆
python -m backend.rl.bc_pretrain --east_data data/east_synthetic.csv

# PPO 热启动（从 SFT 权重开始）
python -m backend.rl.train_rl \
  --n_steps 200000 \
  --warmstart data/rl_models/bc_sft/bc_sft_actor.pt \
  --version v2_sft

# === Phase 3：Offline RL ===
python -m backend.rl.offline_train --use_synthetic

# === Phase 4：World Model + Dyna ===
python -m backend.rl.mbrl_train --use_synthetic --n_iterations 20
```

---

## 名词备注

| 术语 | what（是什么） | why（为什么要有） | how（怎么影响结果） |
|------|--------------|-----------------|------------------|
| **Lawson 准则** | 核聚变点火条件 n·T·τ > 3×10²¹ | 判断等离子体是否有科学价值 | 达成率越高说明控制策略越有效 |
| **q95 安全因子** | 磁场线在 95% 磁通面的绕转圈数 | < 2.0 触发 KS 不稳定性 | 最关键的破裂预判指标 |
| **β_N 归一化 beta** | 等离子体压强 / 磁压 | > 3.5 触发 Troyon 极限 | 控制加热功率和密度防止超限 |
| **τ_E（ITER98pY2）** | 能量约束时间经验公式 | 描述等离子体能量保持能力 | 越大 Lawson 参数越高 |
| **gaming** | Agent 钻奖励设计漏洞刷分 | 检测策略是否真实有效 | 比例 > 0.5 需检查奖励设计 |
| **SFT** | 监督微调 = 行为克隆 | 给 RL 提供有意义的起点 | 减少冷启动探索时间 |
| **CQL** | 保守 Q 学习（离线 RL）| 防止 OOD 动作 Q 值过高估计 | conservative_weight 越大越保守 |
| **Dyna MBRL** | 世界模型 + RL 交替训练 | 用虚拟轨迹提升样本效率 | 不确定性高的区域用真实数据补充 |
| **DTW** | 动态时间规整距离 | 比较两条不等长轨迹的相似度 | 分数 < 10 = 策略有物理意义 |
| **Ensemble** | 多个独立模型的集合 | 估计预测不确定性 | 方差高 = 需要更多数据的区域 |

---

*文档维护：阿策 · 最后更新：2026-02*
