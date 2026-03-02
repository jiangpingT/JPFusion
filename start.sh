#!/bin/bash
# FusionLab 一键启动脚本
# 用法：cd fusion-platform && bash start.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo "  ⚛  FusionLab — 可控核聚变 AI 计算平台"
echo "======================================================"
echo ""

# ── 检查 Python venv ────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo "[1/4] 创建 Python 虚拟环境..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "[2/4] 安装后端依赖..."
pip install -q -r backend/requirements.txt

# ── 创建数据目录 ────────────────────────────────────────────
mkdir -p data

# ── 启动后端 ─────────────────────────────────────────────────
echo "[3/4] 启动后端 FastAPI（端口 8000）..."
uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0 &
BACKEND_PID=$!
echo "  后端 PID: $BACKEND_PID"
echo "  API 文档: http://localhost:8000/docs"

# 等待后端就绪
sleep 2

# ── 启动前端 ─────────────────────────────────────────────────
echo "[4/4] 启动前端 React（端口 3000）..."
cd frontend
npm start &
FRONTEND_PID=$!
echo "  前端 PID: $FRONTEND_PID"
echo "  前端地址: http://localhost:3000"

echo ""
echo "======================================================"
echo "  平台已启动！浏览器访问 http://localhost:3000"
echo "  按 Ctrl+C 停止所有服务"
echo "======================================================"

# 等待任意进程退出
wait $BACKEND_PID $FRONTEND_PID
