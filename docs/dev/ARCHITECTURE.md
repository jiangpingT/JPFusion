# 系统架构说明

> FusionLab 整体技术架构，面向开发者和维护者。

---

## 架构全图

```
┌─────────────────────────────────────────────────────────────────┐
│                      浏览器 (React)                              │
│  ┌─────────────────────┐    ┌───────────────────────────────┐   │
│  │  等离子体推理（MVP）  │    │     FusionRL 控制台           │   │
│  │  PlasmaVisualization│    │     RLDashboard               │   │
│  │  TrainingDashboard  │    │  - 奖励曲线                   │   │
│  │  ParameterPanel     │    │  - Lawson 参数轨迹             │   │
│  └──────────┬──────────┘    └───────────────┬───────────────┘   │
│             │ HTTP/WS                        │ HTTP/WS           │
└─────────────┼────────────────────────────────┼───────────────────┘
              │                                │
┌─────────────┼────────────────────────────────┼───────────────────┐
│             ▼          FastAPI (8000)         ▼                   │
│   /api/train          /api/inference       /api/rl/train         │
│   /api/model-status   /api/plasma-profile  /api/rl/evaluate      │
│   /ws/training        /api/plasma-profile-gs  /api/rl/ws         │
│                                            /api/calibration/*    │
│                                            /api/rl/offline-train │
│                                            /api/rl/mbrl-train    │
├─────────────────────────────────────────────────────────────────┤
│                    后端模块层                                     │
│                                                                   │
│  ┌──────────────┐  ┌──────────────────────────────────────────┐  │
│  │  MVP 路线     │  │           FusionRL 路线                   │  │
│  │              │  │                                           │  │
│  │ physics/     │  │  Phase 1: dynamics → disruption           │  │
│  │ plasma_engine│  │           rewards → fusion_env            │  │
│  │ gs_engine    │  │           train_rl → eval_rl              │  │
│  │              │  │                                           │  │
│  │ ai/          │  │  Phase 2: east_loader → physics_calibrator│  │
│  │ trainer      │  │           strategy_validator → bc_pretrain│  │
│  │ inference    │  │                                           │  │
│  │              │  │  Phase 3: replay_buffer → offline_rl      │  │
│  └──────────────┘  │                                           │  │
│                    │  Phase 4: world_model → world_model_env   │  │
│                    │           mbrl_train                      │  │
│                    └──────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    数据层                                         │
│  data/                                                            │
│  ├── dataset.json      (MVP 物理推理训练数据)                     │
│  ├── plasma_model.pt   (PINN 模型权重)                           │
│  ├── rl_models/        (Phase 1-4 RL 模型)                       │
│  ├── rl_eval/          (评估结果 JSON)                           │
│  └── east/             (ITPA IDDB 数据，用户自行下载)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 两条技术路线

### 路线 A：物理推理（MVP）

```
用户设置参数 (n_e, T_e, B)
    ↓
物理引擎（plasma_engine.py / gs_engine.py）
    ↓ 生成训练数据（高斯解析 + MHD 湍流）
PINN 模型（ai/trainer.py）
    ↓ 训练（MLP + 物理约束损失）
推理（ai/inference.py）
    ↓ 预测等离子体温度分布
前端可视化（PlasmaVisualization）
```

**缺陷**：数据自生自用（高斯生成 → MLP 拟合），无真实科学价值，是 demo 级别闭环。

### 路线 B：FusionRL（四阶段）

```
Phase 1: FusionEnv ODE + PPO → 基础控制策略（无数据）
    ↓
Phase 2: EAST 数据 → 物理校准 + SFT + DTW 验证
    ↓
Phase 3: EAST 数据 → CQL 离线 RL（无需仿真器）
    ↓
Phase 4: EAST 数据 → World Model → Dyna MBRL → 最优策略
```

**价值**：真实物理约束、真实 gaming 问题、真实收敛挑战——科学级别的 RL 实验。

---

## 关键数据流

### Phase 1 训练数据流

```
FusionEnv.reset() → 随机初始状态（合法范围）
    ↓
PPO.learn() → FusionEnv.step(action) × n_steps
    ↓ 每步：step_plasma_state() 更新状态 → check_disruption() → compute_reward()
    ↓ 每 5000 步：FusionRLCallback 评估 + 更新 _rl_training_state
    ↓
API GET /api/rl/status → 前端轮询
WebSocket /api/rl/ws → 实时推送
```

### Phase 2 数据校准流

```
ITPA IDDB (CSV/HDF5)
    ↓ east_loader.load_itpa_iddb()
DataFrame (n_e, T_e, B, Ip, P_heat, q95, tau_E)
    ↓ physics_calibrator.fit_tau_e_coefficients()
τ_E 系数 → 注入 dynamics.TAU_E_COEFFICIENTS
    ↓ FusionEnv 动力学更新
更准确的物理仿真
```

### Phase 4 世界模型流

```
EAST DataFrame
    ↓ replay_buffer.build_replay_buffer()
(obs, actions, rewards, next_obs, terminals)
    ↓ world_model.train_world_model()
WorldModelEnsemble（5 MLP）
    ↓ world_model_env.WorldModelEnv
Gymnasium 接口（供 PPO 训练）
    ↓ PPO.learn() in WorldModelEnv
PPO Agent（Dyna 训练）
    ↓ 在真实 FusionEnv 评估
最终策略
```

---

## WebSocket 实时推送

后端维护两个独立的 WebSocket 连接管理器：

| 管理器 | 端点 | 用途 |
|--------|------|------|
| `manager` | `/ws/training-progress` | MVP PINN 训练曲线 |
| `rl_manager` | `/api/rl/ws` | FusionRL 训练曲线 |

推送格式（FusionRL）：
```json
{
  "type": "rl_progress",
  "timestep": 10000,
  "mean_reward": 12345.6,
  "mean_ep_length": 487,
  "disruption_rate": 0.04,
  "lawson_rate": 0.20,
  "gaming_proxy": 0.012,
  "timestamp": 1740000000.0
}
```

---

## 扩展指南

### 新增 RL 算法

1. 在 `backend/rl/` 下新建 `train_xxx.py`
2. 实现 `train_xxx(n_steps, ...)` 函数 + `get_xxx_state()` 状态查询
3. 在 `backend/main.py` 添加 API 端点
4. 新增 ADR 记录决策

### 新增前端面板

1. 在 `frontend/src/components/` 新建组件目录
2. 在 `frontend/src/App.tsx` 添加 Tab
3. 在 `frontend/src/api/client.ts` 添加 API 调用函数

### 新增破裂条件

1. 在 `backend/rl/disruption.py::check_disruption()` 添加新判断
2. 在 `backend/rl/disruption.py::disruption_margin()` 添加裕度计算
3. 更新 `docs/training/FUSIONRL_TRAINING.md` 中的破裂条件表格

---

*文档维护：阿策 · 最后更新：2026-02*
