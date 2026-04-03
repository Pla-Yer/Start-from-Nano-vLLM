# Start from Nano-vLLM (中文版)

> **[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的中文教育扩展版本**
>
> 本项目在原版 nano-vLLM 基础上，提供了完整的中文文档、实践实验和生产级 API 服务，帮助开发者深入理解 LLM 推理引擎的核心原理。

从零开始构建的轻量级 vLLM 实现，附带教育资源和实用扩展。**所有文档均采用中文撰写，便于中文开发者学习。**

## 关于本项目

本项目基于优秀的 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 实现，并进行了以下扩展：

- **📚 深度中文文档**：全面分析 vLLM 架构，包括调度器、KV Cache、推理流程和分布式推理，**所有文档均为中文撰写**
- **🧪 实践实验套件**：通过动手实验理解批处理策略、加速技术和 KV Cache 优化
- **🌐 生产级 API 服务**：OpenAI 兼容的 FastAPI 服务，支持流式响应、多模型管理和性能监控
- **📖 学习导向设计**：专为教学目的设计，帮助开发者深入理解 LLM 推理引擎内部机制

适用场景：
- 学习现代 LLM 推理引擎的工作原理
- 理解 PagedAttention 和 Continuous Batching 等优化技术
- 实验不同的加速策略
- 使用轻量级推理引擎构建生产应用

## 项目结构

```
nano-vllm/
├── nanovllm/              # 核心推理引擎（原项目）
│   ├── engine/            # LLMEngine, Scheduler, ModelRunner, BlockManager
│   ├── layers/            # Attention, Linear, LayerNorm, RotaryEmbedding
│   ├── models/            # Qwen3 模型实现
│   └── utils/             # 模型加载器和工具函数
├── api/                   # 🆕 FastAPI 服务
│   ├── routes/            # API 端点（chat, completion, metrics）
│   ├── model_manager.py   # 动态模型加载
│   └── router.py          # 请求路由
├── experiments/           # 🆕 实践实验
│   ├── batching/          # 静态 vs 连续批处理
│   ├── acceleration/      # CUDA Graph, Torch compilation
│   └── kv_cache/          # Prefix caching 基准测试
├── docs/                  # 🆕 中文文档
│   ├── 01-architecture.md # 整体架构与请求流程
│   ├── 02-scheduler.md    # 调度器设计
│   ├── 03-kv-cache.md     # KV Cache 管理
│   ├── 04-inference.md    # 推理流程
│   ├── 05-distributed.md  # 分布式推理
│   ├── 大语言模型结构-qwen3.md  # Qwen3 模型结构详解
│   └── nano-vllm.md       # nano-vLLM 实现细节
├── monitoring/            # 🆕 性能监控
└── config/                # 🆕 配置文件
```

## 核心特性

* 🚀 **快速离线推理** - 与 vLLM 相当的推理速度
* 📖 **可读性强的代码库** - 约 1,200 行 Python 代码的清晰实现
* ⚡ **优化套件** - Prefix caching、Tensor Parallelism、Torch compilation、CUDA graph 等
* 🌐 **OpenAI 兼容 API** - 支持流式响应和多模型管理的 FastAPI 服务
* 📚 **完整中文文档** - 详细的架构分析和实现指南
* 🧪 **实验套件** - 理解优化技术的实践实验

## 安装

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## 模型下载

下载模型（qwen3-0.6b）权重：
```bash
python downloadModel.py
```

## 快速开始

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 🌐 API 服务

Nano-vLLM 提供了 OpenAI 兼容的 FastAPI 服务，支持流式响应和多模型管理。

### 启动 API 服务

```bash
python api/main.py
```

服务默认在 `http://localhost:8000` 启动。

### API 端点

| 端点 | 说明 |
|----------|-------------|
| `GET /health` | 健康检查 |
| `GET /v1/models` | 列出可用模型 |
| `POST /v1/chat/completions` | Chat 对话（OpenAI 兼容） |
| `POST /v1/completions` | Text 补全（OpenAI 兼容） |
| `GET /metrics` | 性能指标 |
| `POST /admin/models/{model_name}/load` | 加载模型 |
| `POST /admin/models/{model_name}/unload` | 卸载模型 |

### 使用示例

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Hello, Nano-vLLM!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### 配置

API 服务配置通过 `config/models.yaml` 管理。详见配置文件中的模型路径、GPU 内存利用率和路由策略等选项。

## 📚 文档

`docs/` 目录提供了完整的中文文档，深入解析 vLLM 架构和 nano-vLLM 实现：

### 架构系列

1. **[01-architecture.md](docs/01-architecture.md)** - 整体架构与请求流程
   - PagedAttention 机制
   - Continuous Batching 策略
   - 系统组件交互

2. **[02-scheduler.md](docs/02-scheduler.md)** - 调度器设计与实现
   - 请求调度算法
   - Prefill/Decode 分离
   - 内存管理策略

3. **[03-kv-cache.md](docs/03-kv-cache.md)** - KV Cache 管理
   - 块分配策略
   - Prefix caching 优化
   - 内存效率分析

4. **[04-inference.md](docs/04-inference.md)** - 推理流程详解
   - 逐步推理过程
   - CUDA Graph 优化
   - 张量并行

5. **[05-distributed.md](docs/05-distributed.md)** - 分布式推理
   - 多 GPU 支持
   - 模型并行
   - 通信模式

### 模型分析

- **[大语言模型结构-qwen3.md](docs/大语言模型结构-qwen3.md)** - Qwen3 模型架构详细分析
- **[nano-vllm.md](docs/nano-vllm.md)** - nano-vLLM 实现细节完整解析

## 🧪 实验

`experiments/` 目录包含实践实验，帮助理解关键优化技术：

### 1. 批处理策略 (`experiments/batching/`)

对比静态批处理与连续批处理的性能：

```bash
python experiments/batching/static.py      # 静态批处理
python experiments/batching/continuous.py  # 连续批处理
```

**关键发现：**
- 连续批处理：2278 tok/s
- 静态批处理（最优）：2248 tok/s
- 连续批处理在延迟和吞吐量上全面优于静态批处理

详见 [experiments/batching/README.md](experiments/batching/README.md)。

### 2. 加速技术 (`experiments/acceleration/`)

测试 CUDA Graph 和 Torch compilation：

```bash
python experiments/acceleration/compile.py    # Torch compilation
python experiments/acceleration/cudagraph.py  # CUDA Graph
```

**关键发现：**
- Eager 模式：2883 tok/s
- CUDA Graph 模式：7164 tok/s
- **性能提升：148.5%**

详见 [experiments/acceleration/README.md](experiments/acceleration/README.md)。

### 3. KV Cache 优化 (`experiments/kv_cache/`)

测试 prefix caching 和 KV cache 效率：

```bash
python experiments/kv_cache/prefix.py     # Prefix caching
python experiments/kv_cache/benchmark.py  # KV cache 基准测试
```

详见 [experiments/kv_cache/README.md](experiments/kv_cache/README.md)。

## 性能基准

详见 `bench.py`。

**测试配置：**
- 硬件：RTX 4070 Laptop (8GB)
- 模型：Qwen3-0.6B
- 总请求数：256 个序列
- 输入长度：随机采样 100–1024 tokens
- 输出长度：随机采样 100–1024 tokens

**性能结果：**
| 推理引擎 | 输出 Tokens | 时间 (s) | 吞吐量 (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## 原项目

本项目是 [GeeeekExplorer](https://github.com/GeeeekExplorer) 的 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的中文教育扩展版本。

**新增内容：**
- 📚 `docs/` 目录中的完整中文文档
- 🧪 `experiments/` 目录中的实验套件
- 🌐 `api/` 目录中的 FastAPI 服务
- 📊 `monitoring/` 目录中的性能监控
- ⚙️ `config/` 目录中的配置管理

**核心引擎：** 核心推理引擎（`nanovllm/`）保持与原项目一致，确保相同的优秀性能特性。
