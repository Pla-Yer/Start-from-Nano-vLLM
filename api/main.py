"""
nano-vllm API 服务入口
OpenAI 兼容接口
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yaml
import uvicorn

from api.model_manager import ModelManager
from api.router import ModelRouter
from api.routes import health, models, chat, completion, metrics as metrics_router, admin


# 全局状态
model_manager: ModelManager = None
model_router: ModelRouter = None
metrics = None
gpu_metrics = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global model_manager, model_router, metrics, gpu_metrics

    from monitoring.metrics import MetricsCollector, GPUMetricsCollector

    # 启动时初始化
    config_path = os.path.join(os.path.dirname(__file__), "../config/models.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_manager = ModelManager(config)
    model_router = ModelRouter(config)
    metrics = MetricsCollector()
    gpu_metrics = GPUMetricsCollector()

    # 加载默认模型
    await model_manager.load_model(config["routing"]["default_model"])

    print(f"🚀 nano-vllm API 服务启动完成")
    print(f"   默认模型: {config['routing']['default_model']}")

    yield

    # 关闭时清理
    print("\n🧹 收到退出信号，清理资源...")
    if model_manager:
        model_manager.loaded_models.clear()
    print("👋 API 服务已关闭")


app = FastAPI(
    title="nano-vllm API",
    description="OpenAI-compatible inference API for nano-vllm",
    version="1.0.0",
    lifespan=lifespan,
)


# 注册路由
app.include_router(health.router)
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(completion.router)
app.include_router(metrics_router.router)
app.include_router(admin.router)


# 根路径
@app.get("/")
async def root():
    """根路径 - 返回 HTML 文档"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>nano-vllm API</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }
        .method { display: inline-block; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
        .get { background: #61affe; color: white; }
        .post { background: #49cc90; color: white; }
        code { background: #eee; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>🚀 nano-vllm API</h1>
    <p>OpenAI 兼容的推理 API</p>
    <h2>可用端点</h2>
    <div class="endpoint"><span class="method get">GET</span> <code>/health</code> - 健康检查</div>
    <div class="endpoint"><span class="method get">GET</span> <code>/v1/models</code> - 模型列表</div>
    <div class="endpoint"><span class="method post">POST</span> <code>/v1/chat/completions</code> - Chat 对话</div>
    <div class="endpoint"><span class="method post">POST</span> <code>/v1/completions</code> - Text Completion</div>
    <div class="endpoint"><span class="method get">GET</span> <code>/metrics</code> - 性能指标</div>
    <h2>示例</h2>
    <pre><code>curl http://localhost:8000/v1/models</code></pre>
</body>
</html>"""
    return HTMLResponse(content=html)


# Embedding 占位符
@app.post("/v1/embeddings")
async def embeddings(request):
    """Embedding 接口 (占位符，当前不支持)"""
    from fastapi import HTTPException
    raise HTTPException(
        status_code=501,
        detail="Embeddings not yet supported. Only chat/completion available."
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )