#!/bin/bash
# FusionLab 一键启动脚本（外网访问版）
# 访问地址：http://<IP>:18791/jpfusion

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

# ── 构建前端 ─────────────────────────────────────────────────
echo "[3/4] 构建前端（npm run build）..."
cd frontend
PUBLIC_URL=/jpfusion npm run build
cd ..
echo "  前端构建完成 → frontend/build/"

# ── 启动后端（同时承担前端静态文件服务）─────────────────────
echo "[4/4] 启动 FastAPI（端口 18791，0.0.0.0）..."
uvicorn backend.main:app --port 18791 --host 0.0.0.0

echo ""
echo "======================================================"
echo "  平台已启动！"
echo "  本地访问：http://localhost:18791/jpfusion"
echo "  外网访问：http://<你的公网IP>:18791/jpfusion"
echo "  API 文档：http://localhost:18791/docs"
echo "  按 Ctrl+C 停止"
echo "======================================================"
