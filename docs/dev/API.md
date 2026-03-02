# API 文档

> 完整 API 列表。也可访问 `http://localhost:8000/docs` 查看 Swagger UI（含在线调试）。

---

## 基础信息

- **Base URL**：`http://localhost:8000`
- **Content-Type**：`application/json`
- **WebSocket**：`ws://localhost:8000/api/rl/ws`

---

## Phase 1 — FusionRL RL 训练

### POST /api/rl/train

启动 PPO RL 训练（后台异步运行）。

**请求体**

```json
{
  "n_steps": 200000,
  "n_envs": 4,
  "learning_rate": 0.0003,
  "ent_coef": 0.005,
  "max_ep_steps": 500,
  "warmstart_path": "",
  "version": ""
}
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `n_steps` | int | 200000 | 总训练步数（10000 ~ 5000000）|
| `n_envs` | int | 4 | 并行环境数（1 ~ 16）|
| `learning_rate` | float | 3e-4 | PPO 学习率 |
| `ent_coef` | float | 0.005 | 熵系数（探索强度，越大越随机）|
| `max_ep_steps` | int | 500 | 每 episode 最大步数 |
| `warmstart_path` | string | "" | SFT BC 预训练权重路径（空=冷启动）|
| `version` | string | "" | 版本标签（用于模型目录命名）|

**响应**

```json
{
  "message": "RL 训练已启动（后台运行）",
  "n_steps": 200000,
  "n_envs": 4,
  "ws_url": "ws://localhost:8000/api/rl/ws"
}
```

---

### GET /api/rl/status

查询 RL 训练进度。

**响应**

```json
{
  "status": "training",
  "total_steps": 45000,
  "mean_reward": 18234.5,
  "mean_ep_length": 487.2,
  "disruption_rate": 0.04,
  "lawson_rate": 0.20,
  "elapsed_s": 312.4,
  "model_path": null,
  "error_msg": null
}
```

| 字段 | 说明 |
|------|------|
| `status` | idle / training / done / error |
| `disruption_rate` | 破裂率（目标 < 0.1）|
| `lawson_rate` | Lawson 准则达成率（> 0 有科学价值）|

---

### GET /api/rl/history

获取训练历史曲线（最近 200 条评估记录）。

**响应**

```json
{
  "history": [
    {
      "timestep": 5000,
      "mean_reward": 8234.1,
      "mean_ep_length": 423.0,
      "disruption_rate": 0.20,
      "lawson_rate": 0.00,
      "gaming_proxy": 0.034,
      "timestamp": 1740000000.0
    }
  ]
}
```

---

### POST /api/rl/evaluate

运行模型评估（同步，约 20-60 秒）。

**请求体**

```json
{
  "model_path": "data/rl_models/ppo_fusion_v1/final.zip",
  "n_episodes": 10,
  "max_steps": 500
}
```

**响应**

```json
{
  "success": true,
  "summary": {
    "n_episodes": 10,
    "mean_reward": 18234.5,
    "std_reward": 2341.1,
    "mean_ep_length": 487.2,
    "disruption_rate": 0.04,
    "lawson_rate": 0.30,
    "gaming_proxy": 0.012,
    "lawson_target": 3e21,
    "model_path": "data/rl_models/..."
  }
}
```

---

### GET /api/rl/trajectory

获取最新评估轨迹（供前端绘图）。需先调用 `/api/rl/evaluate`。

**响应**（结构）

```json
{
  "episodes": [
    {
      "trajectory": {
        "steps":   [0, 1, 2, ...],
        "n_e":     [3.1e19, 3.2e19, ...],
        "T_e":     [1.2e7, 1.3e7, ...],
        "q95":     [3.2, 3.1, ...],
        "beta_N":  [0.8, 0.9, ...],
        "lawson":  [1.2e20, 1.4e20, ...],
        "rewards": [-5.2, -4.8, ...]
      },
      "stats": {
        "episode": 0,
        "total_reward": 18234.5,
        "n_steps": 487,
        "disrupted": false,
        "lawson_achieved": true,
        "final_lawson": 4.2e21
      }
    }
  ],
  "summary": { ... }
}
```

---

### GET /api/rl/models

扫描所有已训练模型版本，返回元数据列表（按最佳奖励降序）。

**响应**

```json
{
  "models": [
    {
      "version": "v5",
      "final_path": "data/rl_models/ppo_fusion_v5/final.zip",
      "checkpoints": ["...80000_steps.zip", "...final.zip"],
      "best_reward": 187047.7,
      "last_reward": 144035.4,
      "lawson_rate": 0.80,
      "n_records": 60,
      "latest_ckpt": "data/rl_models/ppo_fusion_v5/final.zip"
    }
  ]
}
```

---

### POST /api/rl/infer/live

**推理能力 1**：在最新检查点（按修改时间）上跑 1 个确定性 episode，用于实时观察训练中的模型能力。

**响应**（同 `/api/rl/infer/best`）

---

### POST /api/rl/infer/best

**推理能力 2**：在历史最佳模型（按 best_reward 排序）上跑 1 个 episode。可通过 query param 指定模型路径。

**Query 参数**（可选）

| 参数 | 类型 | 说明 |
|------|------|------|
| model_path | string | 指定模型 `.zip` 路径，不传则自动选最佳 |

**响应**

```json
{
  "trajectory": {
    "step": [0, 1, ...],
    "n_e": [...], "T_e": [...], "q95": [...],
    "beta_N": [...], "Ip": [...], "P_heat": [...],
    "lawson": [...], "tau_E": [...], "reward": [...]
  },
  "total_reward": 125842.3,
  "n_steps": 500,
  "disrupted": false,
  "lawson_achieved": true,
  "final_lawson": 1.2e27,
  "version": "v5",
  "source": "best",
  "lawson_target": 1e27
}
```

---

### POST /api/rl/infer/demo

**推理能力 3**：自生数据规则控制策略演示，无需已训练模型。使用启发式规则（梯度推进 P_heat）演示等离子体演化过程，可作为 RL 策略对比基线。

**响应**（同上，附加 `"policy": "rule-based（梯度推进 P_heat，无模型）"`）

---

### WS /api/rl/ws

实时接收训练曲线推送。

连接后立即收到初始化消息：
```json
{
  "type": "rl_init",
  "status": "training",
  "history": [...]
}
```

训练过程中每 5000 步收到：
```json
{
  "type": "rl_progress",
  "timestep": 10000,
  "mean_reward": 12345.6,
  "disruption_rate": 0.08,
  "lawson_rate": 0.10,
  "gaming_proxy": 0.023
}
```

---

## Phase 2 — 数据校准

### POST /api/rl/bc-pretrain

SFT 行为克隆预训练（后台异步）。

**请求体**

```json
{
  "east_data_path": "data/east_synthetic.csv",
  "n_epochs": 500,
  "lr": 0.001
}
```

**响应**

```json
{
  "message": "SFT 行为克隆预训练已启动（后台运行）"
}
```

---

### POST /api/calibration/fit-tau-e

用 EAST 数据拟合 τ_E 系数。

**请求体**

```json
{
  "data_path": "data/east_synthetic.csv"
}
```

**响应**

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
  "r_squared": 0.87,
  "n_samples": 19800,
  "message": "拟合成功（R²=0.8700）"
}
```

---

### GET /api/calibration/tau-e-coeff

查询当前 τ_E 系数（ITER98pY2 默认值或已校准值）。

---

### POST /api/calibration/validate

DTW 策略有效性验证。

**请求体**

```json
{
  "model_path": "data/rl_models/ppo_fusion_v1/final.zip",
  "east_data_path": "data/east_synthetic.csv",
  "n_episodes": 5
}
```

**响应**

```json
{
  "success": true,
  "dtw_score": 8.4,
  "dtw_scores": [7.2, 8.9, 9.1, 7.8, 9.0],
  "judgment": "有物理意义（DTW 距离与 EAST 专家接近）",
  "agent_traj_length": 487.0,
  "east_traj_length": 200
}
```

---

## Phase 3 — Offline RL

### POST /api/rl/offline-train

启动 CQL 离线 RL 训练（后台异步）。

**请求体**

```json
{
  "east_data_path": "data/east_synthetic.csv",
  "n_steps": 100000,
  "conservative_weight": 5.0
}
```

### GET /api/rl/offline-status

```json
{
  "status": "training",
  "steps": 45000,
  "loss": null,
  "model_path": null
}
```

---

## Phase 4 — World Model + Dyna

### POST /api/rl/train-world-model

**请求体**

```json
{
  "east_data_path": "data/east_synthetic.csv",
  "n_epochs": 100,
  "n_models": 5
}
```

### POST /api/rl/mbrl-train

**请求体**

```json
{
  "east_data_path": "data/east_synthetic.csv",
  "n_iterations": 50,
  "n_rl_steps_per_iter": 10000
}
```

### GET /api/rl/world-model-uncertainty

返回 (n_e_norm, P_heat_norm) 二维不确定性热图（20×20 网格）。

**响应**

```json
{
  "success": true,
  "n_e_axis": [0.0, 0.053, ...],
  "P_heat_axis": [0.0, 0.053, ...],
  "uncertainty": [[0.001, 0.002, ...], ...],
  "min_uncertainty": 0.001,
  "max_uncertainty": 0.089
}
```

---

## MVP 路线 API（已有）

| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/generate-data` | POST | 生成物理推理训练数据集 |
| `/api/train` | POST | 启动 PINN 训练 |
| `/ws/training-progress` | WS | PINN 训练实时进度 |
| `/api/inference` | POST | 等离子体温度分布推理 |
| `/api/model-status` | GET | PINN 模型状态 |
| `/api/plasma-profile` | POST | 物理解析剖面（不经过 AI）|
| `/api/plasma-profile-gs` | POST | GS 方程求解剖面 |

---

*文档维护：阿策 · 最后更新：2026-02*
