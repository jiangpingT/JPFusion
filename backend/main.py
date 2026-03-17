"""
main.py — FastAPI 应用入口

API 路由：
  POST /api/generate-data      — 用物理引擎生成训练数据集
  POST /api/train              — 启动异步训练
  WS   /ws/training-progress   — WebSocket 实时训练指标推送
  POST /api/inference          — 推理，返回分布图数据
  GET  /api/model-status       — 查询模型和训练状态
  GET  /api/training-history   — 获取历史训练曲线数据

  四阶段命名体系（数据来源 × 学习方式）：
    Phase 1 · Sim-RL     仿真环境在线强化学习（PPO）
    Phase 2 · Sim-SFT    仿真环境监督微调，行为克隆（BC）→ Phase 1 热启动
    Phase 3 · Offline-RL 历史数据离线强化学习（CQL）← Phase 1 轨迹
    Phase 4 · Model-RL   世界模型强化学习（MBRL），← Phase 3 策略初始化

  — Phase 1 · Sim-RL —
  POST /api/rl/train           — 启动 PPO 在线训练（FusionEnv）
  GET  /api/rl/status          — 查询训练进度
  POST /api/rl/evaluate        — 运行评估（返回轨迹）
  GET  /api/rl/trajectory      — 最新评估轨迹
  WS   /api/rl/ws              — 实时推送训练曲线

  — Phase 2 · Sim-SFT —
  POST /api/calibration/fit-tau-e    — 拟合 τ_E 系数（2A）
  GET  /api/calibration/tau-e-coeff  — 查询当前系数
  POST /api/calibration/validate     — DTW 验证策略（2B）
  POST /api/rl/bc-pretrain           — 行为克隆 SFT 预训练（2C）

  — Phase 3 · Offline-RL —
  POST /api/rl/offline-train   — 启动 CQL 离线训练
  GET  /api/rl/offline-status  — 查询 Offline-RL 进度

  — Phase 4 · Model-RL —
  POST /api/rl/train-world-model        — 训练世界模型（MLP Ensemble）
  POST /api/rl/mbrl-train               — 启动 Dyna MBRL 循环
  GET  /api/rl/world-model-uncertainty  — 世界模型不确定性热图

启动命令：
  cd fusion-platform
  uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import json
import os
import numpy as np
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── 本地模块 ────────────────────────────────────────────────────────────────
from backend.physics.plasma_engine import (
    generate_training_dataset,
    generate_plasma_profile_2d,
    compute_scalar_physics,
)
from backend.ai.trainer import (
    train_model,
    get_training_state,
    restore_state_from_db,
    _training_state,
)
from backend.ai.inference import run_inference, reload_model

# ── 数据路径 ─────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), "../data")
DATASET_PATH  = os.path.join(DATA_DIR, "dataset.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ── FastAPI 应用 ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="FusionLab API",
    description="可控核聚变 AI 计算平台 — 后端 API",
    version="2.0.0",
)

# ── 启动事件：从 SQLite 恢复训练状态 ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    restore_state_from_db()
    print("[FusionLab] 后端已启动，训练状态已从 SQLite 恢复")

# ── CORS（同源部署，放开所有 origin）──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 前端静态文件（npm run build 产物）────────────────────────────────────────
_FRONTEND_BUILD = os.path.join(os.path.dirname(__file__), "../frontend/build")

if os.path.isdir(_FRONTEND_BUILD):
    # /static/  →  前端 JS / CSS 等静态资源（gateway 剥掉 /jpfusion 前缀后对应此路径）
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(_FRONTEND_BUILD, "static")),
        name="frontend-static",
    )

    # 其他根目录静态文件（favicon.ico / manifest.json 等）
    for _f in ("favicon.ico", "manifest.json", "robots.txt", "logo192.png", "logo512.png"):
        _fp = os.path.join(_FRONTEND_BUILD, _f)
        if os.path.exists(_fp):
            _path = f"/{_f}"
            app.mount(_path, StaticFiles(directory=_FRONTEND_BUILD, html=False), name=f"static-{_f}")

    # SPA catch-all：所有非 API 路由返回 index.html（必须在所有 API 路由之后注册）
    @app.get("/", include_in_schema=False)
    async def serve_spa_root():
        return FileResponse(os.path.join(_FRONTEND_BUILD, "index.html"))

# ── WebSocket 连接管理器 ─────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: dict):
        """广播到所有连接的客户端"""
        msg = json.dumps(data)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)


manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic 请求/响应模型
# ─────────────────────────────────────────────────────────────────────────────

class GenerateDataRequest(BaseModel):
    n_samples:      int  = Field(default=500,  ge=10,   le=5000,  description="参数空间采样数")
    grid_size:      int  = Field(default=8,    ge=4,    le=32,    description="空间网格分辨率")
    add_turbulence: bool = Field(default=True,                    description="是否叠加 MHD 湍流扰动")


class TrainRequest(BaseModel):
    n_epochs:    int   = Field(default=100,   ge=5,    le=1000)
    batch_size:  int   = Field(default=512,   ge=32,   le=4096)
    lr:          float = Field(default=1e-3,  ge=1e-5, le=1e-1)
    max_samples: int   = Field(default=30000, ge=100,  le=200000)
    # PINN 物理约束参数
    use_pinn:    bool  = Field(default=True,                     description="启用 PINN 物理约束损失")
    lambda_pde:  float = Field(default=0.01,  ge=0.0,  le=1.0,  description="PDE 残差权重")
    lambda_mono: float = Field(default=0.05,  ge=0.0,  le=1.0,  description="单调性约束权重")
    lambda_sym:  float = Field(default=0.02,  ge=0.0,  le=1.0,  description="方位角对称权重")
    lambda_bc:   float = Field(default=0.03,  ge=0.0,  le=1.0,  description="边界条件权重")


class InferenceRequest(BaseModel):
    n_e:        float = Field(default=1e19,  description="电子密度 (m^-3)")
    T_e:        float = Field(default=1e7,   description="电子温度 (K)")
    B:          float = Field(default=5.0,   description="磁场强度 (T)")
    grid_size:  int   = Field(default=32,    ge=8, le=64)


# ─────────────────────────────────────────────────────────────────────────────
#  API 路由
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "FusionLab API 运行中",
        "version": "2.0.0",
        "endpoints": [
            "POST /api/generate-data",
            "POST /api/train",
            "WS   /ws/training-progress",
            "POST /api/inference",
            "GET  /api/model-status",
            "GET  /api/training-history",
        ]
    }


@app.post("/api/generate-data")
async def generate_data(req: GenerateDataRequest, background_tasks: BackgroundTasks):
    """
    生成训练数据集

    用 PlasmaPy 物理引擎在参数空间扫描，生成 (n_e, T_e, B, r, theta) → T 的训练样本
    add_turbulence=True 时叠加 MHD 湍流扰动，使数据更真实
    数据保存到 data/dataset.json
    """
    if _training_state["status"] == "training":
        raise HTTPException(status_code=409, detail="训练正在进行中，无法重新生成数据")

    _training_state["status"] = "generating"

    def _do_generate():
        try:
            dataset = generate_training_dataset(
                n_samples=req.n_samples,
                grid_size=req.grid_size,
                save_path=DATASET_PATH,
                add_turbulence=req.add_turbulence,
            )
            _training_state["status"] = "idle"
            return len(dataset)
        except Exception as e:
            _training_state["status"] = "error"
            _training_state["error_msg"] = str(e)
            raise

    # 在后台运行（生成可能需要几十秒）
    background_tasks.add_task(_do_generate)

    estimated = req.n_samples * req.grid_size * req.grid_size
    return {
        "message":           "数据集生成已启动（后台运行）",
        "n_params_samples":  req.n_samples,
        "grid_size":         req.grid_size,
        "add_turbulence":    req.add_turbulence,
        "estimated_samples": estimated,
        "save_path":         DATASET_PATH,
    }


@app.get("/api/generate-data/status")
async def generate_data_status():
    """查询数据集生成状态"""
    exists = os.path.exists(DATASET_PATH)
    size   = os.path.getsize(DATASET_PATH) if exists else 0
    return {
        "dataset_exists":  exists,
        "dataset_size_mb": round(size / 1024 / 1024, 2),
        "status":          _training_state["status"],
    }


@app.post("/api/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    启动模型训练（异步后台任务）

    训练过程中的指标通过 WebSocket /ws/training-progress 实时推送
    use_pinn=True 时启用 PINN 物理约束损失
    """
    if _training_state["status"] == "training":
        raise HTTPException(status_code=409, detail="已有训练任务在运行中")

    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=400, detail="数据集不存在，请先调用 /api/generate-data")

    async def _ws_push(metrics: dict):
        await manager.broadcast(metrics)

    async def _train():
        await train_model(
            n_epochs=req.n_epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            ws_callback=_ws_push,
            max_data_samples=req.max_samples,
            use_pinn=req.use_pinn,
            lambda_pde=req.lambda_pde,
            lambda_mono=req.lambda_mono,
            lambda_sym=req.lambda_sym,
            lambda_bc=req.lambda_bc,
        )
        # 训练完成后重新加载推理模型缓存
        reload_model()

    background_tasks.add_task(_train)

    return {
        "message":    "训练已启动",
        "n_epochs":   req.n_epochs,
        "batch_size": req.batch_size,
        "lr":         req.lr,
        "use_pinn":   req.use_pinn,
        "ws_url":     "/ws/training-progress",
    }


@app.websocket("/ws/training-progress")
async def ws_training_progress(websocket: WebSocket):
    """
    WebSocket 端点：实时接收训练进度

    客户端连接后，每个 epoch 结束会收到一条 JSON：
    {
      "type": "training_progress",
      "epoch": 15,
      "total_epochs": 100,
      "train_loss": 0.0023,
      "val_loss": 0.0031,
      "pde_loss": 0.0005,
      ...
    }
    连接后立即推送当前状态，方便页面刷新后恢复
    """
    await manager.connect(websocket)
    # 连接时推送当前状态快照
    state = get_training_state()
    await websocket.send_text(json.dumps({
        "type":         "init",
        "status":       state["status"],
        "epoch":        state["epoch"],
        "total_epochs": state["total_epochs"],
        "history":      state["history"][-50:],   # 最近50条历史
    }))
    try:
        while True:
            # 保持连接，等待客户端关闭
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/api/inference")
async def inference(req: InferenceRequest):
    """
    推理接口

    给定 (n_e, T_e, B)，在截面网格上计算温度分布
    若模型已训练则用 AI 推理，否则用物理解析解（后备模式）

    返回 Plotly heatmap 所需格式
    """
    try:
        result = run_inference(
            n_e=req.n_e,
            T_e=req.T_e,
            B=req.B,
            grid_size=req.grid_size,
            use_physics_fallback=True,
        )
        return {
            "success": True,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-status")
async def model_status():
    """查询模型状态和训练信息"""
    from backend.ai.inference import get_model
    import os

    model_path = os.path.join(DATA_DIR, "plasma_model.pt")
    model_exists   = os.path.exists(model_path)
    dataset_exists = os.path.exists(DATASET_PATH)

    state = get_training_state()

    best_val_loss = state["best_val_loss"]
    # float("inf") 不可 JSON 序列化，转为 None
    if best_val_loss == float("inf"):
        best_val_loss = None

    return {
        "model_ready":     model_exists and get_model() is not None,
        "model_exists":    model_exists,
        "dataset_exists":  dataset_exists,
        "training_status": state["status"],
        "current_epoch":   state["epoch"],
        "total_epochs":    state["total_epochs"],
        "train_loss":      state["train_loss"],
        "val_loss":        state["val_loss"],
        "best_val_loss":   best_val_loss,
        "error_msg":       state["error_msg"],
    }


@app.get("/api/training-history")
async def training_history():
    """
    获取完整训练历史曲线数据（用于前端页面刷新后恢复显示）

    返回：最近 200 条 epoch 记录
    """
    state = get_training_state()
    return {
        "status":  state["status"],
        "history": state["history"][-200:],
        "epoch":   state["epoch"],
        "total_epochs": state["total_epochs"],
    }


@app.post("/api/plasma-profile-gs")
async def plasma_profile_gs(req: InferenceRequest):
    """
    Grad-Shafranov 平衡等离子体剖面（物理增强版）

    求解 G-S 方程得到真实磁通面坐标 ψ_N(R,Z)，
    在磁通面坐标系中建立温度剖面（比高斯模型物理上准确一个量级）。

    返回额外字段：psiN_profile（磁通面分布），gs_meta（平衡参数）
    耗时约 1-3 秒（Picard 迭代）
    """
    try:
        from backend.physics.gs_engine import generate_plasma_profile_2d_gs
        from backend.physics.plasma_engine import compute_scalar_physics

        profile = generate_plasma_profile_2d_gs(
            n_e=req.n_e,
            T_e=req.T_e,
            B=req.B,
            grid_size=req.grid_size,
        )
        physics = compute_scalar_physics(req.n_e, req.T_e, req.B)
        T_arr = np.array(profile["T_profile"])
        return {
            "source":        "physics_gs",
            "T_values":      profile["T_profile"],
            "psiN_profile":  profile["psiN_profile"],
            "n_profile":     profile["n_profile"],
            "r_grid":        profile["r_grid"],
            "theta_grid":    profile["theta_grid"],
            "x_grid":        profile["x_grid"],
            "y_grid":        profile["y_grid"],
            "T_min":         float(T_arr.min()),
            "T_max":         float(T_arr.max()),
            "grid_size":     req.grid_size,
            "physics_params": {
                "lambda_D":  float(physics["lambda_D"]),
                "omega_p":   float(physics["omega_p"]),
                "v_alfven":  float(physics["v_alfven"]),
                "beta":      float(physics["beta"]),
            },
            "gs_meta":       profile["gs_meta"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GS 求解失败: {e}")


@app.post("/api/plasma-profile")
async def plasma_profile(req: InferenceRequest):
    """
    直接用物理引擎（不经过 AI）计算等离子体剖面
    用于基准对比
    """
    profile = generate_plasma_profile_2d(
        n_e=req.n_e,
        T_e=req.T_e,
        B=req.B,
        grid_size=req.grid_size,
    )
    physics = compute_scalar_physics(req.n_e, req.T_e, req.B)
    return {
        "source":        "physics_engine",
        **profile,
        "physics_params": physics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 · Sim-RL — 仿真环境在线强化学习（PPO）
#  数据来源：FusionEnv 实时仿真  |  算法：stable-baselines3 PPO
# ═══════════════════════════════════════════════════════════════════════════════

from backend.rl.train_rl import train_rl, get_rl_training_state
from backend.rl.eval_rl  import evaluate_model, get_latest_trajectory


class RLTrainRequest(BaseModel):
    n_steps:       int   = Field(default=200_000, ge=10_000, le=5_000_000)
    n_envs:        int   = Field(default=4,  ge=1, le=16)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1e-2)
    ent_coef:      float = Field(default=0.005, ge=0.0, le=0.1)
    max_ep_steps:  int   = Field(default=500, ge=50, le=2000)
    warmstart_path: str  = Field(default="", description="SFT BC 预训练权重路径（留空则冷启动）")
    version:       str   = Field(default="", description="版本标签")


class RLEvalRequest(BaseModel):
    model_path:  str = Field(description="模型 zip 路径")
    n_episodes:  int = Field(default=10, ge=1, le=100)
    max_steps:   int = Field(default=500, ge=50, le=2000)


# ─── WebSocket 连接管理（RL 专用）────────────────────────────────────────────
class RLConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data, default=float)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)


rl_manager = RLConnectionManager()


@app.post("/api/rl/train")
async def rl_train(req: RLTrainRequest, background_tasks: BackgroundTasks):
    """启动 PPO RL 训练（后台任务）"""
    state = get_rl_training_state()
    if state["status"] == "training":
        raise HTTPException(status_code=409, detail="RL 训练已在运行中")

    async def _ws_push(metrics: dict):
        await rl_manager.broadcast(metrics)

    def _do_train():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run():
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: train_rl(
                    n_steps=req.n_steps,
                    n_envs=req.n_envs,
                    learning_rate=req.learning_rate,
                    ent_coef=req.ent_coef,
                    max_ep_steps=req.max_ep_steps,
                    warmstart_path=req.warmstart_path or None,
                    version=req.version or None,
                )
            )

        loop.run_until_complete(_run())
        loop.close()

    background_tasks.add_task(
        train_rl,
        n_steps=req.n_steps,
        n_envs=req.n_envs,
        learning_rate=req.learning_rate,
        ent_coef=req.ent_coef,
        max_ep_steps=req.max_ep_steps,
        warmstart_path=req.warmstart_path or None,
        version=req.version or None,
    )

    return {
        "message":    "RL 训练已启动（后台运行）",
        "n_steps":    req.n_steps,
        "n_envs":     req.n_envs,
        "ws_url":     "/api/rl/ws",
    }


@app.get("/api/rl/status")
async def rl_status():
    """查询 RL 训练进度"""
    state = get_rl_training_state()
    # float("inf") 不可 JSON 序列化
    return {k: (None if v == float("inf") else v)
            for k, v in state.items()
            if k != "history"}


@app.get("/api/rl/history")
async def rl_history():
    """获取 RL 训练历史曲线"""
    state = get_rl_training_state()
    return {"history": state["history"][-200:]}


@app.post("/api/rl/evaluate")
async def rl_evaluate(req: RLEvalRequest):
    """运行评估，返回状态轨迹（同步，可能需要 20-30 秒）"""
    if not os.path.exists(req.model_path):
        raise HTTPException(status_code=404, detail=f"模型不存在: {req.model_path}")
    try:
        result = evaluate_model(
            model_path=req.model_path,
            n_episodes=req.n_episodes,
            max_steps=req.max_steps,
        )
        return {"success": True, "summary": result["summary"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/trajectory")
async def rl_trajectory():
    """获取最新评估轨迹（供前端绘图）"""
    traj = get_latest_trajectory()
    if not traj:
        raise HTTPException(status_code=404, detail="暂无评估数据，请先调用 /api/rl/evaluate")
    return traj


@app.websocket("/api/rl/ws")
async def rl_ws(websocket: WebSocket):
    """RL 训练实时曲线 WebSocket"""
    await rl_manager.connect(websocket)
    state = get_rl_training_state()
    await websocket.send_text(json.dumps({
        "type":    "rl_init",
        "status":  state["status"],
        "history": state["history"][-50:],
    }, default=float))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        rl_manager.disconnect(websocket)


# ─── 推理能力 API（三能力）──────────────────────────────────────────────────────

def _scan_models(phase_filter: str | None = None) -> list[dict]:
    """扫描所有 ppo_fusion_* 模型目录，返回元数据列表（按最佳奖励降序）

    phase 识别规则（优先级由高到低）：
      1. 目录内 phase_tag.txt 文件内容（如 "phase2"）
      2. 目录名含 "bc" / "sft" / "phase2" → phase2
      3. 其余 → phase1

    Args:
        phase_filter: "phase1" / "phase2" / None（不过滤）
    """
    models_dir = Path(__file__).parent.parent / "data" / "rl_models"
    result = []
    if not models_dir.exists():
        return result

    for d in sorted(models_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("p1_ppo_"):
            continue
        version = d.name.replace("p1_ppo_", "")
        hist_path  = d / "history.json"
        final_path = d / "final.zip"
        tag_path   = d / "phase_tag.txt"

        # 确定所属 phase
        if tag_path.exists():
            phase = tag_path.read_text().strip()
        elif any(k in d.name.lower() for k in ("bc", "sft", "phase2")):
            phase = "phase2"
        else:
            phase = "phase1"

        if phase_filter and phase != phase_filter:
            continue

        checkpoints = sorted(
            [str(p) for p in d.glob("*.zip")],
            key=lambda p: os.path.getmtime(p),
        )

        best_reward = None
        last_reward = None
        lawson_rate = None
        n_records   = 0

        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            if hist:
                n_records   = len(hist)
                last        = hist[-1]
                best        = max(hist, key=lambda x: x["mean_reward"])
                best_reward = best["mean_reward"]
                last_reward = last["mean_reward"]
                lawson_rate = last.get("lawson_rate", 0.0)

        result.append({
            "version":      version,
            "phase":        phase,
            "dir":          str(d),
            "final_path":   str(final_path) if final_path.exists() else None,
            "checkpoints":  checkpoints,
            "best_reward":  best_reward,
            "last_reward":  last_reward,
            "lawson_rate":  lawson_rate,
            "n_records":    n_records,
            "latest_ckpt":  checkpoints[-1] if checkpoints else None,
        })

    result.sort(key=lambda m: (m["best_reward"] or -1e9), reverse=True)
    return result


def _run_single_episode(model_path: str, max_steps: int = 500) -> dict:
    """加载模型，跑 1 个确定性 episode，返回轨迹 dict"""
    from stable_baselines3 import PPO
    from backend.rl.fusion_env import FusionEnv
    from backend.rl.dynamics import denormalize_state, compute_tau_E
    from backend.rl.rewards import compute_lawson_parameter, LAWSON_TARGET

    model = PPO.load(model_path)
    env   = FusionEnv(max_steps=max_steps)
    obs, _ = env.reset()
    done   = False

    traj = {k: [] for k in ["step", "n_e", "T_e", "q95", "beta_N", "Ip", "P_heat", "lawson", "tau_E", "reward"]}
    ep_reward  = 0.0
    disrupted  = False
    lawson_ok  = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        s = denormalize_state(obs)
        tau_E  = compute_tau_E(s["n_e"], s["B"], s["P_heat"], s["Ip"])
        lawson = compute_lawson_parameter(s["n_e"], s["T_e"], tau_E)

        traj["step"].append(step)
        traj["n_e"].append(float(s["n_e"]))
        traj["T_e"].append(float(s["T_e"]))
        traj["q95"].append(float(s["q95"]))
        traj["beta_N"].append(float(s["beta_N"]))
        traj["Ip"].append(float(s["Ip"]))
        traj["P_heat"].append(float(s["P_heat"]))
        traj["lawson"].append(float(lawson))
        traj["tau_E"].append(float(tau_E))
        traj["reward"].append(float(rew))

        ep_reward += rew
        if info.get("disrupted"):
            disrupted = True
        if info.get("lawson_achieved"):
            lawson_ok = True
        step += 1

    return {
        "trajectory":      traj,
        "total_reward":    float(ep_reward),
        "n_steps":         step,
        "disrupted":       disrupted,
        "lawson_achieved": lawson_ok,
        "final_lawson":    traj["lawson"][-1] if traj["lawson"] else 0.0,
        "model_path":      model_path,
        "lawson_target":   1e27,
    }


def _run_demo_episode(max_steps: int = 500) -> dict:
    """规则控制策略（无需模型）演示等离子体演化"""
    import numpy as np
    from backend.rl.fusion_env import FusionEnv
    from backend.rl.dynamics import denormalize_state, compute_tau_E
    from backend.rl.rewards import compute_lawson_parameter

    env = FusionEnv(max_steps=max_steps)
    obs, _ = env.reset()
    done = False

    traj = {k: [] for k in ["step", "n_e", "T_e", "q95", "beta_N", "Ip", "P_heat", "lawson", "tau_E", "reward"]}
    ep_reward = 0.0
    disrupted = False
    lawson_ok = False
    step = 0

    while not done:
        # 规则策略：梯度推进 P_heat，稳定 Ip，控制密度
        n_e, T_e, B, q95, beta_N, Ip, P_heat = obs
        delta_P = 0.02 if P_heat < 0.65 else (-0.015 if beta_N > 0.6 else 0.005)
        delta_Ip = 0.01 if Ip < 0.55 else -0.005
        delta_n  = 0.008 if n_e < 0.45 else (-0.01 if n_e > 0.65 else 0.001)
        action = np.array([delta_P, delta_n, delta_Ip], dtype=np.float32)
        action = np.clip(action, -0.1, 0.1)

        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        s = denormalize_state(obs)
        tau_E  = compute_tau_E(s["n_e"], s["B"], s["P_heat"], s["Ip"])
        lawson = compute_lawson_parameter(s["n_e"], s["T_e"], tau_E)

        traj["step"].append(step)
        traj["n_e"].append(float(s["n_e"]))
        traj["T_e"].append(float(s["T_e"]))
        traj["q95"].append(float(s["q95"]))
        traj["beta_N"].append(float(s["beta_N"]))
        traj["Ip"].append(float(s["Ip"]))
        traj["P_heat"].append(float(s["P_heat"]))
        traj["lawson"].append(float(lawson))
        traj["tau_E"].append(float(tau_E))
        traj["reward"].append(float(rew))

        ep_reward += rew
        if info.get("disrupted"):
            disrupted = True
        if info.get("lawson_achieved"):
            lawson_ok = True
        step += 1

    return {
        "trajectory":      traj,
        "total_reward":    float(ep_reward),
        "n_steps":         step,
        "disrupted":       disrupted,
        "lawson_achieved": lawson_ok,
        "final_lawson":    traj["lawson"][-1] if traj["lawson"] else 0.0,
        "policy":          "rule-based（梯度推进 P_heat，无模型）",
        "lawson_target":   1e27,
    }


def _run_offline_episode(model_path: str, max_steps: int = 500) -> dict:
    """使用 d3rlpy CQL 模型跑 1 个确定性 episode，返回轨迹 dict"""
    import d3rlpy
    from backend.rl.fusion_env import FusionEnv
    from backend.rl.dynamics import denormalize_state, compute_tau_E
    from backend.rl.rewards import compute_lawson_parameter

    policy = d3rlpy.load_learnable(model_path)
    env    = FusionEnv(max_steps=max_steps)
    obs, _ = env.reset()
    done   = False

    traj = {k: [] for k in ["step", "n_e", "T_e", "q95", "beta_N", "Ip", "P_heat", "lawson", "tau_E", "reward"]}
    ep_reward = 0.0
    disrupted = False
    lawson_ok = False
    step = 0

    while not done:
        action = policy.predict(obs.reshape(1, -1))[0]
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        s = denormalize_state(obs)
        tau_E  = compute_tau_E(s["n_e"], s["B"], s["P_heat"], s["Ip"])
        lawson = compute_lawson_parameter(s["n_e"], s["T_e"], tau_E)

        traj["step"].append(step)
        traj["n_e"].append(float(s["n_e"]))
        traj["T_e"].append(float(s["T_e"]))
        traj["q95"].append(float(s["q95"]))
        traj["beta_N"].append(float(s["beta_N"]))
        traj["Ip"].append(float(s["Ip"]))
        traj["P_heat"].append(float(s["P_heat"]))
        traj["lawson"].append(float(lawson))
        traj["tau_E"].append(float(tau_E))
        traj["reward"].append(float(rew))

        ep_reward += rew
        if info.get("disrupted"):
            disrupted = True
        if info.get("lawson_achieved"):
            lawson_ok = True
        step += 1

    return {
        "trajectory":      traj,
        "total_reward":    float(ep_reward),
        "n_steps":         step,
        "disrupted":       disrupted,
        "lawson_achieved": lawson_ok,
        "final_lawson":    traj["lawson"][-1] if traj["lawson"] else 0.0,
        "model_path":      model_path,
        "lawson_target":   1e27,
    }


@app.get("/api/rl/models")
async def rl_models():
    """扫描所有已训练模型版本，返回元数据列表（按最佳奖励降序）"""
    return {"models": _scan_models()}


@app.post("/api/rl/infer/live")
async def rl_infer_live():
    """推理能力 1：在最新检查点上跑 1 个 episode（实时训练模型推理）"""
    models = _scan_models()
    if not models:
        raise HTTPException(status_code=404, detail="暂无已训练模型，请先启动训练")

    # 找到所有 checkpoints 中最新修改的一个
    all_ckpts = []
    for m in models:
        for ck in m["checkpoints"]:
            all_ckpts.append((os.path.getmtime(ck), ck, m["version"]))
    all_ckpts.sort(reverse=True)

    if not all_ckpts:
        raise HTTPException(status_code=404, detail="暂无检查点文件")

    _, latest_ckpt, version = all_ckpts[0]
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _run_single_episode(latest_ckpt)
        )
        result["source"]  = "live"
        result["version"] = version
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rl/infer/best")
async def rl_infer_best(model_path: str = None):
    """推理能力 2：在最佳模型上跑 1 个 episode（若传入 model_path 则用指定模型）"""
    if model_path and os.path.exists(model_path):
        target_path = model_path
        version = "custom"
    else:
        models = _scan_models()
        if not models:
            raise HTTPException(status_code=404, detail="暂无已训练模型")
        best = models[0]  # 已按 best_reward 降序
        target_path = best["final_path"] or best["latest_ckpt"]
        version = best["version"]
        if not target_path:
            raise HTTPException(status_code=404, detail="最佳模型无可用文件")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _run_single_episode(target_path)
        )
        result["source"]  = "best"
        result["version"] = version
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rl/infer/demo")
async def rl_infer_demo():
    """推理能力 3：自生数据推理 Demo（规则控制策略，无需模型）"""
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_demo_episode
        )
        result["source"] = "demo"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rl/infer/offline")
async def rl_infer_offline():
    """Phase 3 CQL 推理：加载 CQL 离线模型跑 1 个 episode"""
    models_dir = Path(__file__).parent.parent / "data" / "rl_models"
    cql_dir    = models_dir / "p3_cql"
    # 优先选专家版本（p3_cql_expert.d3），其次任意 .d3，最后 .pt
    _expert = cql_dir / "p3_cql_expert.d3"
    cql_model = (
        _expert if cql_dir.exists() and _expert.exists()
        else (next(cql_dir.glob("*.d3"), None)
              or next(cql_dir.glob("*.zip"), None)
              or next(cql_dir.glob("*.pt"),  None))
        if cql_dir.exists() else None
    )

    if not cql_model:
        raise HTTPException(status_code=404, detail="未找到 CQL 模型，请先完成 Phase 3 离线训练")

    cql_path = str(cql_model)
    try:
        if cql_model.suffix == ".zip":
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _run_single_episode(cql_path)
            )
        else:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _run_offline_episode(cql_path)
            )
        result["source"]  = "offline-cql"
        result["version"] = f"cql-{cql_model.stem}"
        return result
    except ImportError:
        raise HTTPException(status_code=500, detail="d3rlpy 未安装，无法推理 CQL 模型")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/phase-status")
async def rl_phase_status():
    """查询四阶段训练状态 — Sim-RL / Sim-SFT / Offline-RL / Model-RL 各自的模型准备情况"""
    models_dir = Path(__file__).parent.parent / "data" / "rl_models"

    # ── Phase 1 · Sim-RL：PPO 在线强化学习（仅 phase1 标记的模型）─────────
    p1_models = _scan_models(phase_filter="phase1")
    phase1 = {
        "ready":       len(p1_models) > 0,
        "models":      p1_models,
        "best_model":  p1_models[0]["final_path"] if p1_models else None,
        "best_reward": p1_models[0]["best_reward"] if p1_models else None,
        "best_lawson": p1_models[0]["lawson_rate"] if p1_models else None,
        "n_versions":  len(p1_models),
    }

    # ── Phase 2 · Sim-SFT：行为克隆 SFT Actor + BC 热启动 Phase 1 ────────
    sft_path  = models_dir / "p2_sft" / "p2_sft_actor.pt"
    p2_models = _scan_models(phase_filter="phase2")
    # Phase 2 PPO 最佳模型（按 best_reward）
    p2_best   = p2_models[0] if p2_models else None
    phase2 = {
        "sft_ready":           sft_path.exists(),
        "sft_path":            str(sft_path) if sft_path.exists() else None,
        "sft_size_mb":         round(sft_path.stat().st_size / 1e6, 2) if sft_path.exists() else None,
        "ppo_warmstart_ready": len(p2_models) > 0,
        "ppo_warmstart_path":  p2_best["final_path"] or p2_best["latest_ckpt"] if p2_best else None,
        "ppo_models":          p2_models,
        "demo_model":          (p2_best["final_path"] or p2_best["latest_ckpt"]) if p2_best else None,
        "best_reward":         p2_best["best_reward"] if p2_best else None,
        "best_lawson":         p2_best["lawson_rate"] if p2_best else None,
    }

    # ── Phase 3 · Offline-RL：CQL 离线强化学习（← Phase 1 轨迹数据）────────
    cql_dir   = models_dir / "p3_cql"
    # 优先选专家轨迹版本（p3_cql_expert.d3），其次任意 .d3，最后兼容旧 .pt
    def _pick_cql_model(d):
        if not d.exists():
            return None
        expert = d / "p3_cql_expert.d3"
        if expert.exists():
            return expert
        return next(d.glob("*.d3"), None) or next(d.glob("*.pt"), None)
    cql_model = _pick_cql_model(cql_dir)
    phase3 = {
        "ready":      cql_model is not None,
        "cql_path":   str(cql_model) if cql_model else None,
        "needs_east": True,  # CQL 需要 EAST 真实数据
    }

    # ── Phase 4 · Model-RL：世界模型 + Dyna MBRL（← Phase 3 策略初始化）──
    # 优先用 v2（专家世界模型+精调），其次回退到 v1
    wm_v2_path = models_dir / "world_model" / "world_model_v2.pt"
    wm_path    = wm_v2_path if wm_v2_path.exists() else models_dir / "world_model" / "world_model.pt"
    # MBRL 模型：v3 优先（pipeline v2 产出），其次 v2/v1 回退
    def _pick_mbrl_model(base):
        for cand_name in ["p4_mbrl_v3_finetuned.zip", "p4_mbrl_v3_best.zip",
                          "p4_mbrl_finetuned.zip", "p4_mbrl_final.zip", "p4_mbrl_best.zip"]:
            p = base / "p4_mbrl" / cand_name
            if p.exists():
                return p
        v2_any = next((base / "p4_mbrl").glob("*.zip"), None) if (base / "p4_mbrl").exists() else None
        if v2_any:
            return v2_any
        # 回退：p4_mbrl_v1/ 目录中最优（final 优先，否则 best，否则任意）
        mbrl_old = base / "p4_mbrl_v1"
        if not mbrl_old.exists():
            return None
        for name in ["mbrl_agent_final.zip", "mbrl_agent_best.zip"]:
            p = mbrl_old / name
            if p.exists():
                return p
        return next(mbrl_old.glob("*.zip"), None)
    mbrl_model = _pick_mbrl_model(models_dir)

    # 读取 v3 评估结果（pipeline v2 写入的 JSON）
    v3_results_path = models_dir / "p4_mbrl" / "v3_results.json"
    v3_results = {}
    if v3_results_path.exists():
        try:
            with open(v3_results_path) as f:
                v3_results = json.load(f)
        except Exception:
            pass

    # 读取冠军记录（champion.json — 精调后由 phase4_finetune.py 写入，始终最新最优）
    champion_path = models_dir / "p4_mbrl" / "champion.json"
    champion_data = {}
    if champion_path.exists():
        try:
            with open(champion_path) as f:
                champion_data = json.load(f)
        except Exception:
            pass

    # 取 champion.json 与 v3_results.json 中的最大值，确保 Dashboard 显示真实最高分
    v3_reward   = v3_results.get("mean_reward") or -1e9
    champ_reward = champion_data.get("mean_reward") or -1e9
    if champ_reward > v3_reward:
        best_reward  = champion_data.get("mean_reward")
        best_std     = champion_data.get("std_reward")
        best_lawson  = champion_data.get("lawson_rate")
        disrupt_rate = champion_data.get("disrupt_rate")
    else:
        best_reward  = v3_results.get("mean_reward")
        best_std     = v3_results.get("std_reward")
        best_lawson  = v3_results.get("lawson_rate")
        disrupt_rate = v3_results.get("disrupt_rate")

    phase4 = {
        "world_model_ready": wm_path.exists(),
        "world_model_path":  str(wm_path) if wm_path.exists() else None,
        "world_model_size_mb": round(wm_path.stat().st_size / 1e6, 2) if wm_path.exists() else None,
        "mbrl_ready":        mbrl_model is not None,
        "mbrl_path":         str(mbrl_model) if mbrl_model else None,
        "best_reward":       best_reward,
        "best_std":          best_std,
        "best_lawson":       best_lawson,
        "disrupt_rate":      disrupt_rate,
        "version":           champion_data.get("archive") or v3_results.get("version"),
        "notes":             v3_results.get("notes"),
    }

    return {
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "phase4": phase4,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 · Sim-SFT — 仿真环境监督微调（行为克隆 BC）
#  数据来源：专家轨迹（v5 / EAST）  |  算法：MLP 监督学习 → Phase 1 热启动
# ═══════════════════════════════════════════════════════════════════════════════

class TauEFitRequest(BaseModel):
    data_path: str = Field(description="ITPA IDDB 数据文件路径（CSV/HDF5）")


class ValidateRequest(BaseModel):
    model_path:     str = Field(description="RL 模型路径")
    east_data_path: str = Field(description="EAST 放电数据路径")
    n_episodes:     int = Field(default=5, ge=1, le=20)


class BCPretrainRequest(BaseModel):
    east_data_path: str = Field(description="EAST 专家轨迹数据路径")
    n_epochs:       int = Field(default=500, ge=10, le=5000)
    lr:             float = Field(default=1e-3, ge=1e-5, le=1e-1)


@app.post("/api/calibration/fit-tau-e")
async def calibration_fit_tau_e(req: TauEFitRequest, background_tasks: BackgroundTasks):
    """用 EAST 数据拟合 τ_E 系数（Phase 2A）"""
    if not os.path.exists(req.data_path):
        raise HTTPException(status_code=404, detail=f"数据文件不存在: {req.data_path}")
    try:
        from backend.data.east_loader import load_itpa_iddb
        from backend.calibration.physics_calibrator import fit_tau_e_coefficients
        df = load_itpa_iddb(req.data_path)
        coeffs = fit_tau_e_coefficients(df)
        return {"success": True, "coefficients": coeffs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/calibration/tau-e-coeff")
async def calibration_tau_e_coeff():
    """查询当前 τ_E 系数"""
    from backend.rl.dynamics import TAU_E_COEFFICIENTS
    return {"coefficients": TAU_E_COEFFICIENTS}


@app.post("/api/calibration/validate")
async def calibration_validate(req: ValidateRequest):
    """DTW 验证 RL Agent 策略（Phase 2B）"""
    from backend.calibration.strategy_validator import compare_with_east
    from backend.data.east_loader import load_itpa_iddb
    try:
        df = load_itpa_iddb(req.east_data_path)
        result = compare_with_east(req.model_path, df, n_episodes=req.n_episodes)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rl/bc-pretrain")
async def rl_bc_pretrain(req: BCPretrainRequest, background_tasks: BackgroundTasks):
    """SFT 行为克隆预训练（Phase 2C）—— 从 EAST 专家轨迹学习初始策略"""
    if not os.path.exists(req.east_data_path):
        raise HTTPException(status_code=404, detail=f"数据文件不存在: {req.east_data_path}")
    from backend.rl.bc_pretrain import behavior_clone_sft
    background_tasks.add_task(
        behavior_clone_sft,
        east_data_path=req.east_data_path,
        n_epochs=req.n_epochs,
        lr=req.lr,
    )
    return {"message": "SFT 行为克隆预训练已启动（后台运行）"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3 · Offline-RL — 历史数据离线强化学习（CQL）
#  数据来源：Phase 1 最佳轨迹 / EAST 历史数据  |  算法：d3rlpy CQL
# ═══════════════════════════════════════════════════════════════════════════════

# 离线训练状态
_offline_rl_state = {"status": "idle", "steps": 0, "loss": None, "model_path": None}


class OfflineTrainRequest(BaseModel):
    east_data_path: str = Field(description="EAST 数据路径")
    n_steps:        int = Field(default=100_000, ge=10_000, le=1_000_000)
    conservative_weight: float = Field(default=5.0, ge=0.1, le=20.0)


@app.post("/api/rl/offline-train")
async def rl_offline_train(req: OfflineTrainRequest, background_tasks: BackgroundTasks):
    """启动 CQL 离线强化学习训练（Phase 3）"""
    if not os.path.exists(req.east_data_path):
        raise HTTPException(status_code=404, detail=f"数据文件不存在: {req.east_data_path}")
    from backend.rl.offline_rl import train_cql
    background_tasks.add_task(
        train_cql,
        east_data_path=req.east_data_path,
        n_steps=req.n_steps,
        conservative_weight=req.conservative_weight,
        state_dict=_offline_rl_state,
    )
    return {"message": "CQL 离线训练已启动（后台运行）"}


@app.get("/api/rl/offline-status")
async def rl_offline_status():
    """查询 Offline RL 训练进度"""
    return _offline_rl_state


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 4 · Model-RL — 世界模型强化学习（Dyna MBRL）
#  数据来源：Phase 3 最优策略（初始 actor）+ 世界模型虚拟轨迹  |  算法：MLP Ensemble + PPO
# ═══════════════════════════════════════════════════════════════════════════════

_mbrl_state = {"status": "idle", "wm_loss": None, "rl_reward": None, "iteration": 0}


class WorldModelTrainRequest(BaseModel):
    east_data_path: str = Field(description="EAST 数据路径")
    n_epochs:       int = Field(default=100, ge=10, le=1000)
    n_models:       int = Field(default=5, ge=2, le=10, description="Ensemble 模型数量")


class MBRLTrainRequest(BaseModel):
    east_data_path: str = Field(description="EAST 数据路径（世界模型更新数据源）")
    n_iterations:   int = Field(default=50, ge=5, le=500)
    n_rl_steps_per_iter: int = Field(default=10_000, ge=1000, le=100_000)


@app.post("/api/rl/train-world-model")
async def rl_train_world_model(req: WorldModelTrainRequest, background_tasks: BackgroundTasks):
    """从 EAST 数据训练世界模型（Phase 4）"""
    if not os.path.exists(req.east_data_path):
        raise HTTPException(status_code=404, detail=f"数据文件不存在: {req.east_data_path}")
    from backend.rl.world_model import train_world_model
    background_tasks.add_task(
        train_world_model,
        east_data_path=req.east_data_path,
        n_epochs=req.n_epochs,
        n_models=req.n_models,
    )
    return {"message": "世界模型训练已启动（后台运行）"}


@app.post("/api/rl/mbrl-train")
async def rl_mbrl_train(req: MBRLTrainRequest, background_tasks: BackgroundTasks):
    """启动 Dyna MBRL 训练（Phase 4）"""
    if not os.path.exists(req.east_data_path):
        raise HTTPException(status_code=404, detail=f"数据文件不存在: {req.east_data_path}")
    from backend.rl.mbrl_train import run_dyna_mbrl
    background_tasks.add_task(
        run_dyna_mbrl,
        east_data_path=req.east_data_path,
        n_iterations=req.n_iterations,
        n_rl_steps_per_iter=req.n_rl_steps_per_iter,
        state_dict=_mbrl_state,
    )
    return {"message": "Dyna MBRL 训练已启动（后台运行）"}


@app.get("/api/rl/world-model-uncertainty")
async def rl_world_model_uncertainty():
    """查询世界模型不确定性热图（Phase 4）"""
    from backend.rl.world_model import get_uncertainty_map
    try:
        result = get_uncertainty_map()
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── SPA catch-all（必须在所有 API 路由之后）─────────────────────────────────
_FRONTEND_BUILD_PATH = os.path.join(os.path.dirname(__file__), "../frontend/build")
if os.path.isdir(_FRONTEND_BUILD_PATH):
    @app.get("/{path:path}", include_in_schema=False)
    async def spa_fallback(path: str):
        return FileResponse(os.path.join(_FRONTEND_BUILD_PATH, "index.html"))
