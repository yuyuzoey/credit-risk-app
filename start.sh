#!/usr/bin/env bash
# ===================================================================
#  信用卡违约风险分析 —— 一键启动（macOS / Linux）
#
#  用法：
#      chmod +x start.sh
#      ./start.sh
#
#  启动后：前端 http://127.0.0.1:8501   后端 http://127.0.0.1:8000/docs
#  Ctrl+C 会同时停掉前后端。
# ===================================================================

set -e
cd "$(dirname "$0")"

# ---- 找 Python：优先项目内 .venv ----
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
else
    PY="python3"
fi

echo "[1/3] 检查依赖 ..."
if ! "$PY" -c "import fastapi, uvicorn, streamlit, requests" 2>/dev/null; then
    echo "      缺少依赖，正在安装 requirements.txt ..."
    "$PY" -m pip install -r requirements.txt
fi

echo "[2/3] 启动后端 http://127.0.0.1:8000 ..."
"$PY" -m uvicorn backend.main:app --port 8000 --host 127.0.0.1 &
BACK_PID=$!
trap 'echo; echo "正在停止后端 ..."; kill $BACK_PID 2>/dev/null || true' EXIT

# 等后端就绪（最多 30 秒）
for i in $(seq 1 30); do
    if "$PY" -c "import requests,sys;sys.exit(0 if requests.get('http://127.0.0.1:8000/health',timeout=2).ok else 1)" 2>/dev/null; then
        echo "      后端已就绪（接口文档 http://127.0.0.1:8000/docs）"
        break
    fi
    if [ "$i" = "30" ]; then
        echo "  [错误] 后端 30 秒内未就绪，请查看上面的报错。"
        exit 1
    fi
    sleep 1
done

echo "[3/3] 启动前端 http://127.0.0.1:8501 ..."
echo
echo "===================================================================="
echo "  前端地址：http://127.0.0.1:8501"
echo "  后端文档：http://127.0.0.1:8000/docs"
echo "  按 Ctrl+C 同时停止前后端。"
echo "===================================================================="
echo

"$PY" -m streamlit run frontend/app.py --server.port 8501
