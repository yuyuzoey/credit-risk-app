@echo off
chcp 65001 >nul
setlocal

REM ===================================================================
REM  信用卡违约风险分析 —— 一键启动（后端 + 前端）
REM
REM  双击运行即可：
REM    1. 后台起 FastAPI 后端（端口 8000）
REM    2. 前台起 Streamlit 前端（端口 8501），并自动打开浏览器
REM
REM  关闭本窗口 = 同时停掉前后端。
REM
REM  如果想用别的 python 环境，把下面的 PY 改掉，或先激活 venv 再运行。
REM ===================================================================

cd /d "%~dp0"

REM ---- 找 Python：优先项目内 .venv ----
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo [1/3] 检查依赖 ...
"%PY%" -c "import fastapi, uvicorn, streamlit, requests" 2>nul
if errorlevel 1 (
    echo.
    echo  缺少依赖，正在安装 requirements.txt ...
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  [错误] 依赖安装失败，请手动运行： "%PY%" -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo [2/3] 启动后端 http://127.0.0.1:8000 ...
start "后端 API (8000)" /min "%PY%" -m uvicorn backend.main:app --port 8000 --host 127.0.0.1

echo       等待后端就绪 ...
set /a _try=0
:wait_backend
set /a _try+=1
"%PY%" -c "import requests,sys;sys.exit(0 if requests.get('http://127.0.0.1:8000/health',timeout=2).ok else 1)" 2>nul
if not errorlevel 1 goto backend_ready
if %_try% geq 30 (
    echo.
    echo  [错误] 后端 30 秒内未就绪。请查看"后端 API (8000)"窗口的报错。
    pause
    exit /b 1
)
"%PY%" -c "import time;time.sleep(1)"
goto wait_backend

:backend_ready
echo       后端已就绪 ^(接口文档 http://127.0.0.1:8000/docs^)

echo [3/3] 启动前端 http://127.0.0.1:8501 ...
echo.
echo ====================================================================
echo   前端地址：http://127.0.0.1:8501
echo   后端文档：http://127.0.0.1:8000/docs
echo   关闭本窗口会同时停止前后端。
echo ====================================================================
echo.

"%PY%" -m streamlit run frontend\app.py --server.port 8501

echo.
echo 前端已停止。后端窗口请自行关闭（标题为"后端 API (8000)"）。
pause
