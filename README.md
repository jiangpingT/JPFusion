# FusionLab — 可控核聚变 AI 平台

> **使命**：用可信的强化学习，让托卡马克等离子体控制从 demo 走向真正的科学深度。

---

## 项目概述

FusionLab 是一个面向可控核聚变研究的 AI 计算平台，分两条技术路线：

| 路线 | 描述 | 状态 |
|------|------|------|
| **物理推理（MVP）** | 高斯/GS 方程生成数据 → PINN 训练 → 等离子体剖面推理 | ✅ 已上线 |
| **FusionRL（四阶段）** | 纯 RL 控制 → EAST 数据校准 → Offline RL → World Model | ✅ 已实现 |

FusionRL 完全类比 JPRobot 后空翻的训练深度——真实的物理约束、真实的 gaming 问题、真实的收敛挑战。

---

## 快速启动

```bash
git clone <repo>
cd fusion-platform
bash start.sh
```

浏览器打开 `http://localhost:3000`，顶部 Tab 切换：
- **等离子体推理** — 物理推理可视化
- **FusionRL 控制** — 强化学习训练监控

---

## 技术栈

### 后端
| 组件 | 技术 |
|------|------|
| API 框架 | FastAPI + uvicorn |
| 在线 RL | stable-baselines3 PPO |
| 离线 RL | d3rlpy CQL |
| 物理仿真 | gymnasium + 自定义 ODE |
| 数据校准 | scipy curve_fit + FastDTW |
| 世界模型 | PyTorch MLP Ensemble |
| 物理引擎 | PlasmaPy + GS 方程求解器 |

### 前端
| 组件 | 技术 |
|------|------|
| 框架 | React + TypeScript |
| 可视化 | Plotly.js |
| 实时推送 | WebSocket |

### 硬件环境
- Apple M4 Pro · 48GB 统一内存
- GPU：MPS（PyTorch Apple Silicon 加速）
- 不支持 CUDA / IsaacGym

---

## 项目结构

```
fusion-platform/
├── backend/
│   ├── main.py              # FastAPI 入口（全量 API 路由）
│   ├── rl/                  # FusionRL 四阶段
│   │   ├── dynamics.py      # ITER98pY2 τ_E ODE 动力学
│   │   ├── disruption.py    # 4 个破裂终止条件
│   │   ├── rewards.py       # Lawson 准则奖励（防 gaming）
│   │   ├── fusion_env.py    # FusionEnv(gymnasium.Env)
│   │   ├── train_rl.py      # PPO 训练脚本
│   │   ├── eval_rl.py       # 评估脚本
│   │   ├── bc_pretrain.py   # SFT 行为克隆预训练
│   │   ├── offline_rl.py    # CQL 离线 RL
│   │   ├── world_model.py   # MLP Ensemble 世界模型
│   │   ├── world_model_env.py # 世界模型 Gymnasium 包装
│   │   └── mbrl_train.py    # Dyna MBRL 主循环
│   ├── calibration/
│   │   ├── physics_calibrator.py  # τ_E 系数拟合
│   │   └── strategy_validator.py  # DTW 策略验证
│   ├── data/
│   │   ├── east_loader.py   # ITPA IDDB 数据加载
│   │   └── replay_buffer.py # Offline RL Replay Buffer
│   ├── physics/             # 物理推理引擎（MVP 路线）
│   ├── ai/                  # PINN 训练器（MVP 路线）
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── RLDashboard/     # FusionRL 监控面板
│       │   ├── TrainingDashboard/
│       │   ├── PlasmaVisualization/
│       │   └── ParameterPanel/
│       └── api/client.ts
├── docs/
│   ├── training/            # 训练操作手册
│   ├── adr/                 # 架构决策记录（ADR）
│   ├── dev/                 # 开发文档
│   └── test/                # 测试文档
├── data/                    # 运行时数据（gitignore）
│   ├── rl_models/           # 训练模型权重
│   └── rl_eval/             # 评估结果
└── start.sh                 # 一键启动脚本
```

---

## API 概览

完整文档见 `http://localhost:8000/docs`（Swagger UI）

| 路由 | 方法 | 描述 |
|------|------|------|
| `/api/rl/train` | POST | 启动 PPO RL 训练 |
| `/api/rl/status` | GET | 查询训练进度 |
| `/api/rl/evaluate` | POST | 运行评估，返回轨迹 |
| `/api/rl/ws` | WS | 实时训练曲线推送 |
| `/api/rl/bc-pretrain` | POST | SFT 行为克隆预训练 |
| `/api/rl/offline-train` | POST | CQL 离线 RL 训练 |
| `/api/rl/train-world-model` | POST | 训练 MLP Ensemble 世界模型 |
| `/api/rl/mbrl-train` | POST | Dyna MBRL 训练 |
| `/api/calibration/fit-tau-e` | POST | 拟合 τ_E 经验公式系数 |
| `/api/calibration/validate` | POST | DTW 策略有效性验证 |

---

## 文档索引

| 文档 | 路径 | 描述 |
|------|------|------|
| 四阶段训练手册 | `docs/training/FUSIONRL_TRAINING.md` | Phase 1-4 完整操作指南 |
| 开发环境搭建 | `docs/dev/SETUP.md` | 本地开发配置 |
| 系统架构 | `docs/dev/ARCHITECTURE.md` | 整体架构设计说明 |
| API 文档 | `docs/dev/API.md` | 全量 API 参数说明 |
| 测试指南 | `docs/test/TESTING.md` | 测试策略与用例 |
| ADR 索引 | `docs/adr/README.md` | 架构决策记录总览 |

---

## 物理背景（极简版）

| 概念 | 说明 |
|------|------|
| **托卡马克（Tokamak）** | 用螺线管磁场约束等离子体的核聚变装置，EAST 是中国版本 |
| **Lawson 准则** | 核聚变点火条件：n·T·τ_E > 3×10²¹（密度×温度×能量约束时间） |
| **q95 安全因子** | 磁场线绕转圈数，< 2.0 触发 Kruskal-Shafranov 不稳定性（破裂） |
| **β_N（归一化 beta）** | 等离子体压强 / 磁压，> 3.5 触发 Troyon 极限（MHD 破裂） |
| **Greenwald 密度极限** | 最大允许密度，超过后辐射损失暴增导致骤冷破裂 |
| **τ_E（ITER98pY2）** | 能量约束时间经验公式，决定等离子体能量保持多久 |

---

*阿策负责架构管理 · FusionLab v2.0 · 2026-02*
