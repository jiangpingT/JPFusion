# 开发环境搭建

> 本文档面向需要在本地开发和调试 FusionLab 的开发者。

---

## 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | macOS Darwin 24.6.0（Apple Silicon）/ Linux |
| Python | 3.11 ~ 3.12 |
| Node.js | 18+（前端）|
| 内存 | >= 16 GB（推荐 48 GB）|
| GPU | 可选（Apple MPS / NVIDIA CUDA）|

> 本项目在 Apple M4 Pro 48GB 上开发，不依赖 CUDA。

---

## 快速启动（一键）

```bash
git clone <repo>
cd fusion-platform
bash start.sh
```

`start.sh` 会自动：
1. 创建 Python venv（如不存在）
2. 安装后端依赖
3. 启动后端 FastAPI（:8000）
4. 启动前端 React（:3000）

---

## 手动分步启动

### 后端

```bash
cd fusion-platform

# 创建虚拟环境（首次）
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r backend/requirements.txt

# 启动（开发模式，热重载）
uvicorn backend.main:app --reload --port 8000

# API 文档
open http://localhost:8000/docs
```

### 前端

```bash
cd fusion-platform/frontend

# 安装依赖（首次）
npm install

# 启动开发服务器
npm start

# 浏览器自动打开 http://localhost:3000
```

---

## 依赖说明

### 后端依赖（backend/requirements.txt）

| 包 | 版本 | 用途 |
|----|------|------|
| fastapi | 0.115.5 | REST API 框架 |
| uvicorn[standard] | 0.32.1 | ASGI 服务器（WebSocket 支持）|
| pydantic | 2.10.3 | 请求/响应数据验证 |
| numpy | 2.0.2 | 数值计算 |
| torch | 2.5.1 | MLP Ensemble 世界模型（MPS 加速）|
| plasmapy | 2024.5.0 | 等离子体物理量计算（MVP 路线）|
| **gymnasium** | **1.0.0** | FusionEnv 框架（d3rlpy 要求此版本）|
| **stable-baselines3** | **2.7.1** | PPO 在线 RL |
| **d3rlpy** | **2.8.1** | CQL 离线 RL |
| **scipy** | latest | τ_E 系数拟合 |
| **fastdtw** | **0.3.4** | DTW 轨迹距离 |
| **pandas** | >=2.0.0 | EAST 数据处理 |

> **注意**：gymnasium 版本被 d3rlpy 固定为 1.0.0（非最新的 1.2.3），
> 安装时会看到降级提示，属正常现象。

### 前端依赖

| 包 | 用途 |
|----|------|
| react + typescript | UI 框架 |
| plotly.js | 等离子体剖面可视化 |
| axios | HTTP 客户端 |

---

## 项目结构（开发视角）

```
fusion-platform/
├── backend/
│   ├── main.py              # 所有 API 路由的入口，改路由在这里
│   ├── rl/                  # FusionRL 核心（Phase 1-4）
│   ├── calibration/         # EAST 数据校准（Phase 2）
│   ├── data/                # 数据加载和处理
│   ├── physics/             # 物理推理（MVP，不动）
│   └── ai/                  # PINN 训练（MVP，不动）
├── frontend/
│   └── src/
│       ├── App.tsx           # 顶部 Tab 切换，新增 Tab 在这里
│       ├── components/
│       │   ├── RLDashboard/ # FusionRL 监控（改前端在这里）
│       │   └── ...
│       └── api/client.ts    # 后端 API 调用封装
├── docs/                    # 文档（本目录）
│   ├── adr/                 # 架构决策记录
│   ├── dev/                 # 开发文档（本文件所在）
│   ├── test/                # 测试文档
│   └── training/            # 训练操作手册
├── data/                    # 运行时数据（.gitignore）
│   ├── rl_models/           # RL 训练模型
│   └── rl_eval/             # 评估结果
└── start.sh
```

---

## 环境变量

目前不需要环境变量配置。后端默认在 `localhost:8000`，前端默认连接 `localhost:8000`。

如需修改，编辑：
- 后端端口：`start.sh` 中的 `uvicorn ... --port 8000`
- 前端 API 地址：`frontend/src/api/client.ts` 中的 `BASE_URL`

---

## 常见问题

**Q：pip install 时提示 gymnasium 降级**
A：正常，d3rlpy 2.8.1 依赖 gymnasium 1.0.0。

**Q：`RuntimeError: Tensor for argument input is on cpu but expected on mps`**
A：world_model.py 的 predict_next 需要自动检测设备，已在代码中修复。
如仍出现，检查模型是否在 MPS 设备上，但输入未移动到 MPS。

**Q：前端显示 `WebSocket 连接失败`**
A：RL 训练未启动时 WebSocket 连接会失败，属正常现象。点击"训练 PPO RL"后才会有数据推送。

**Q：`import gym` 警告（Gym has been unmaintained...）**
A：d3rlpy 内部使用旧版 gym，警告不影响运行，无需处理。

---

## 开发工作流

```bash
# 1. 新建分支
git checkout -b feature/xxx

# 2. 改代码
# ...

# 3. 运行测试
source venv/bin/activate
python -m pytest docs/test/ -v  # 暂无完整测试套件，见 docs/test/TESTING.md

# 4. 快速验证后端导入
python -c "from backend.rl.fusion_env import FusionEnv; env = FusionEnv(); obs, _ = env.reset(); print('OK', obs)"

# 5. Commit
git add .
git commit -m "feat: ..."
```

---

*文档维护：阿策 · 最后更新：2026-02*
