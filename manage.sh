#!/bin/bash
# ============================================================
# Hermes 后端服务管理脚本
# 用法: ./manage.sh {start|stop|restart|status|logs}
#
# 启动完整的 Hermes Web UI 后端 (端口 9119)，
# 包含所有 /api/* 接口 + 静态前端资源。
# 开发时配合 web/ 目录的 npm run dev 使用。
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.server.pid"
LOG_FILE="$PROJECT_DIR/.server.log"
TOKEN_FILE="$PROJECT_DIR/.server.token"
HOST="127.0.0.1"
PORT=9119

# 找 Python >= 3.11
find_python() {
    if command -v conda &>/dev/null; then
        local conda_py
        conda_py=$(conda run -n base which python 2>/dev/null) || true
        if [ -n "$conda_py" ] && "$conda_py" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
            echo "$conda_py"; return
        fi
    fi
    if [ -x "/opt/homebrew/bin/python3" ]; then
        if /opt/homebrew/bin/python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
            echo "/opt/homebrew/bin/python3"; return
        fi
    fi
    if command -v python3 &>/dev/null; then
        if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
            echo "$(command -v python3)"; return
        fi
    fi
    echo ""
}

SYSTEM_PYTHON=$(find_python)
if [ -z "$SYSTEM_PYTHON" ]; then
    echo "错误: 需要 Python >= 3.11"
    exit 1
fi

echo "Python: $SYSTEM_PYTHON ($($SYSTEM_PYTHON --version))"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON="$VENV_DIR/bin/python"

# 初始化 venv + 安装 web 依赖
init_venv() {
    if [ ! -f "$PYTHON" ]; then
        echo ">>> 创建 venv ..."
        "$SYSTEM_PYTHON" -m venv "$VENV_DIR"
        "$PYTHON" -m pip install --upgrade pip -q
        echo ">>> 安装依赖 ..."
        "$PYTHON" -m pip install -e "$PROJECT_DIR[web]" 2>&1 | tail -3
        echo ">>> venv 初始化完成"
    fi
}

# 启动 Hermes web server
start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "服务已在运行, PID=$(cat "$PID_FILE"), 端口=$PORT"
        return 0
    fi
    init_venv
    echo ">>> 启动 Hermes Web UI (端口 $PORT)..."
    cd "$PROJECT_DIR"
    nohup env HERMES_WEB_DIST="$PROJECT_DIR/hermes_cli/web_dist" "$VENV_DIR/bin/hermes" dashboard --host "$HOST" --port "$PORT" --no-open > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 3
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "服务已启动, PID=$(cat "$PID_FILE")"
        if [ -f "$TOKEN_FILE" ]; then
            echo "Token: $(head -c 8 "$TOKEN_FILE")..."
        fi
    else
        echo "启动失败, 日志:"
        cat "$LOG_FILE"
        return 1
    fi
}

stop() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo ">>> 停止服务, PID=$(cat "$PID_FILE")..."
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        sleep 1
        kill -9 "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE" "$TOKEN_FILE"
        echo "服务已停止"
    else
        echo "服务未在运行"
        rm -f "$PID_FILE" "$TOKEN_FILE"
    fi
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "运行中, PID=$(cat "$PID_FILE"), 端口=$PORT"
        echo "URL: http://$HOST:$PORT"
        [ -f "$TOKEN_FILE" ] && echo "Token: $(head -c 8 "$TOKEN_FILE")..."
    else
        echo "未运行"
    fi
}

logs() {
    [ -f "$LOG_FILE" ] && tail -f "$LOG_FILE" || echo "暂无日志"
}

case "${1:-start}" in
    start)   start   ;;
    stop)    stop    ;;
    restart) stop && sleep 1 && start ;;
    status)  status  ;;
    logs)    logs    ;;
    *)       echo "用法: $0 {start|stop|restart|status|logs}" ;;
esac
