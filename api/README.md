# nano-vllm API

OpenAI 兼容的推理 API 服务。

## 快速开始

### 启动服务

```bash
python api/main.py
```

服务将在 `http://localhost:8000` 启动。

### 使用 curl

```bash
# 健康检查
curl http://localhost:8000/health

# 列出模型
curl http://localhost:8000/v1/models

# Chat 对话
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'

# 流式对话
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true,
    "max_tokens": 100
  }'
```

### 使用 OpenAI 库

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-dummy",
    base_url="http://localhost:8000/v1"
)

# 普通调用
response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100
)
print(response.choices[0].message.content)

# 流式调用
for chunk in client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
    max_tokens=100
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | API 欢迎页面 |
| GET | `/health` | 健康检查 |
| GET | `/v1/models` | 列出可用模型 |
| POST | `/v1/chat/completions` | Chat 对话 |
| POST | `/v1/completions` | Text Completion |
| GET | `/metrics` | 性能指标 |
| GET | `/metrics/gpu` | GPU 状态 |

## 客户端

### 对话客户端

```bash
python api/chat_client.py
```

支持命令：
- `/models` - 列出模型
- `/switch <model>` - 切换模型
- `/clear` - 清空对话历史
- `/exit` - 退出

### 测试脚本

```bash
python api/test_client.py
```

## 配置

修改 `config/models.yaml` 配置模型：

```yaml
models:
  qwen3-0.6b:
    path: /path/to/Qwen3-0.6b
    enabled: true
    max_model_len: 4096

routing:
  default_model: qwen3-0.6b
```

## 目录结构

```
api/
├── main.py           # FastAPI 服务入口
├── schemas.py        # Pydantic 模型定义
├── utils.py          # 工具函数
├── model_manager.py  # 模型加载/卸载
├── router.py         # 模型路由
├── routes/           # API 路由模块
│   ├── health.py     # 健康检查
│   ├── models.py     # 模型管理
│   ├── chat.py       # Chat 接口
│   ├── completion.py # Completion 接口
│   ├── metrics.py    # 指标
│   └── admin.py      # 管理接口
├── chat_client.py    # 对话客户端
└── test_client.py    # 测试脚本
```

## 依赖

- fastapi
- uvicorn
- sse-starlette
- pydantic
- pyyaml
- openai (客户端)
- requests (测试)