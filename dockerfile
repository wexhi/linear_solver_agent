FROM python:3.12.3

# 设置工作目录
WORKDIR /app

# **确保 .dockerignore 里忽略 .env**
# 先复制 requirements.txt，避免每次都重新安装依赖
COPY requirements.txt /app/

# 仅在 requirements.txt 变化时安装依赖（利用 Docker 缓存机制）
RUN pip install --no-cache-dir -r requirements.txt

# 然后复制整个项目（代码修改后可自动生效）
COPY . /app

# 确保 Docker 容器里有 API_KEY，否则阻止启动
CMD ["sh", "-c", " \
    if [ -z \"$DASHSCOPE_API_KEY\" ] || [ -z \"$LANGSMITH_API_KEY\" ]; then \
    echo '❌ 请在 .env 中填写 API Key'; \
    exit 1; \
    fi; \
    \
    echo '✅ LangGraph 服务器启动中...';\
    echo '🌍 访问 LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://localhost:$PORT'; \
    \
    # 启动 LangGraph 服务器 \
    langgraph dev --host 0.0.0.0 --port $PORT & \
    \
    # 等待服务器启动
    sleep 3; \
    \
    # 自动打开浏览器（适用于 Windows、Linux、macOS)\
    python3 -c \"import webbrowser; webbrowser.open('https://smith.langchain.com/studio/?baseUrl=http://localhost:$PORT')\"; \
    \
    # 让进程保持运行
    wait;\
    "]
