# 分布式推理：Tensor Parallelism 深度解析

## 1. 分布式推理概述

### 1.1 为什么需要分布式推理

大语言模型的参数量巨大，单卡无法容纳：

| 模型 | 参数量 | FP16 显存需求 | 单卡能否容纳 |
|------|--------|---------------|--------------|
| LLaMA-7B | 70 亿 | ~14GB | A10G 可以 |
| LLaMA-70B | 700 亿 | ~140GB | 需要多卡 |
| GPT-175B | 1750 亿 | ~350GB | 需要多机 |

**Tensor Parallelism（张量并行）**：将模型的权重矩阵按维度切分，分布到多个 GPU 上计算。

### 1.2 并行策略对比

| 并行方式 | 切分维度 | 通信模式 | 适用场景 |
|----------|----------|----------|----------|
| **Data Parallelism** | 数据 | AllReduce | 相同模型，多份数据 |
| **Tensor Parallelism** | 模型权重 | AllReduce/AllGather | 大模型，单卡放不下 |
| **Pipeline Parallelism** | 层 | P2P 通信 | 多节点，超深模型 |

nano-vllm 采用 **Tensor Parallelism**，支持 1-8 个 GPU。

---

## 2. Tensor Parallelism 核心原理

### 2.1 矩阵乘法的切分

以一个简单的线性层为例：Y = X × W

```
输入 X: [batch, hidden]
权重 W: [hidden, output]

切分方式：
┌─────────────────────────────────────────┐
│           Column Parallel               │
│  W 被按列切分：W = [W1 | W2 | W3]       │
│  每个 GPU 计算：Yi = X × Wi             │
│  结果需要 AllReduce：Y = Y1 + Y2 + Y3   │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│           Row Parallel                  │
│  W 被按行切分：W = [W1; W2; W3]         │
│  每个 GPU 计算：Yi = X × Wi             │
│  输入需要 AllGather：X = [X1, X2, X3]   │
└─────────────────────────────────────────┘
```

### 2.2 Attention 中的张量并行

```
┌──────────────────────────────────────────────────────────┐
│                    Multi-Head Attention                  │
│                                                          │
│  Q、K、V 投影：Column Parallel                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│  │  Q_pro │    │  K_pro │    │  V_pro │              │
│  └────┬────┘    └────┬────┘    └────┬────┘              │
│       │              │              │                    │
│       ▼              ▼              ▼                    │
│  [Q1,Q2]          [K1,K2]        [V1,V2]                │
│   (GPU 0)         (GPU 0)        (GPU 0)                │
│  [Q3,Q4]          [K3,K4]        [V3,V4]                │
│   (GPU 1)         (GPU 1)        (GPU 1)                │
│                                                          │
│  输出投影：Row Parallel                                   │
│  ┌─────────┐                                             │
│  │ O_pro  │  ──► AllReduce ──► 输出 Y                    │
│  └─────────┘                                             │
└──────────────────────────────────────────────────────────┘
```

每个 GPU 只计算部分 head 的 attention，最后通过 AllReduce 汇总结果。

---

## 3. nano-vllm 线性层实现

### 3.1 基础架构

nano-vllm 提供了多种并行线性层（`nanovllm/layers/linear.py`）：

```python
class LinearBase(nn.Module):
    def __init__(self, input_size, output_size, bias=False, tp_dim=None):
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
```

### 3.2 Column Parallel Linear

用于 QKV 投影和输出投影的第一部分：

```python
class ColumnParallelLinear(LinearBase):
    """按列切分的线性层"""
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        # 输出维度按 tp_size 切分
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param, loaded_weight):
        # 从完整权重中切分出当前 rank 的部分
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

### 3.3 Row Parallel Linear

用于 attention 输出投影和 FFN 最后一层：

```python
class RowParallelLinear(LinearBase):
    """按行切分的线性层"""
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        # 输入维度按 tp_size 切分
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # 汇总各 GPU 的结果
            dist.all_reduce(y)
        return y
```

### 3.4 QKV Parallel Linear

专门用于同时计算 Q、K、V 三个投影：

```python
class QKVParallelLinear(ColumnParallelLinear):
    """同时计算 Q、K、V 投影"""
    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads=None):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 输出 = (num_heads + 2 * num_kv_heads) * head_size
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size)
```

---

## 4. 多进程架构

### 4.1 进程启动

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        config = Config(model, **kwargs)

        # 使用 spawn 模式启动子进程
        ctx = mp.get_context("spawn")

        # 为每个 rank 创建进程（rank 0 在主进程）
        self.ps = []
        self.events = []
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 主进程运行 rank 0
        self.model_runner = ModelRunner(config, 0, self.events)
```

### 4.2 分布式初始化

```python
class ModelRunner:
    def __init__(self, config, rank, event):
        # 初始化 NCCL 通信
        dist.init_process_group(
            "nccl",
            "tcp://localhost:2333",
            world_size=config.tensor_parallel_size,
            rank=rank
        )
        torch.cuda.set_device(rank)
        ...
```

### 4.3 进程间通信

使用 `multiprocessing.shared_memory` 传递请求和结果：

```python
class ModelRunner:
    def __init__(self, ...):
        if world_size > 1:
            if rank == 0:
                # 主进程创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # 工作进程连接共享内存
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 进入等待循环
```

**通信协议**：

```
Rank 0 写入：
┌─────────────────────────────────────────┐
│ 4 bytes: 数据长度 n                      │
│ n bytes: pickle([method_name, *args])   │
└─────────────────────────────────────────┘

Rank N 读取：
1. 等待事件信号
2. 读取长度 n
3. 读取并解析数据
4. 调用对应方法
5. 返回结果
```

---

## 5. 模型切分策略

### 5.1 Qwen3 模型的并行配置

```python
# 在 nanovllm/models/qwen3.py 中
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        # QKV 投影：Column Parallel
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=config.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
        )

        # 输出投影：Row Parallel
        self.o_proj = RowParallelLinear(
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
        )

        # FFN 第一层：Column Parallel
        self.gate_up_proj = MergedColumnParallelLinear(...)

        # FFN 第二层：Row Parallel
        self.down_proj = RowParallelLinear(...)
```

### 5.2 切分后的计算流程

```
输入 X
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  QKV 投影 (Column Parallel)                             │
│  GPU 0: Q1, K1, V1 = X × Wq1, Wk1, Wv1                 │
│  GPU 1: Q2, K2, V2 = X × Wq2, Wk2, Wv2                 │
└─────────────────────────────────────────────────────────┘
    │
    ▼ (各 GPU 持有部分 Q、K、V)
┌─────────────────────────────────────────────────────────┐
│  Attention 计算 (每个 GPU 计算自己的 head)               │
│  GPU 0: O1 = attention(Q1, K1, V1)                     │
│  GPU 1: O2 = attention(Q2, K2, V2)                     │
└─────────────────────────────────────────────────────────┘
    │
    ▼ (各 GPU 持有部分输出)
┌─────────────────────────────────────────────────────────┐
│  输出投影 (Row Parallel) ──► AllReduce                  │
│  GPU 0: Y1 = O1 × Wo1                                   │
│  GPU 1: Y2 = O2 × Wo2                                   │
│  Y = Y1 + Y2 (AllReduce)                                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
输出 Y
```

---

## 6. 性能分析

### 6.1 通信开销

张量并行的主要通信是 AllReduce：

```python
# Row Parallel 中的 AllReduce
def forward(self, x):
    y = F.linear(x, self.weight, self.bias)
    if self.tp_size > 1:
        dist.all_reduce(y)  # 汇总结果
    return y
```

**通信量分析**：

对于一个 [batch, hidden] 的输出：
- 通信量 = batch × hidden × dtype_bytes
- 2 卡：1 次 AllReduce
- 4 卡：1 次 AllReduce（Ring 算法）
- 8 卡：1 次 AllReduce（Ring 算法）

### 6.2 扩展效率

理想情况下：

| GPU 数 | 加速比 | 效率 |
|--------|--------|------|
| 1 | 1x | 100% |
| 2 | 1.9x | 95% |
| 4 | 3.6x | 90% |
| 8 | 7x | 87.5% |

实际效率低于理论值，因为：
- 通信开销
- 负载不均衡
- 同步等待

### 6.3 何时使用张量并行

**适合场景**：
- 单卡放不下模型
- 需要多卡推理
- 延迟要求高（vs pipeline 并行）

**不适合场景**：
- 模型较小，单卡可运行
- 吞吐量优先（考虑 data parallel）
- 多节点（考虑 pipeline parallel）

---

## 7. 与 vLLM 的对比

| 特性 | nano-vllm | vLLM |
|------|------------|------|
| TP 实现 | 基础 NCCL | 更完善的 NCCL 优化 |
| 通信方式 | SharedMemory | 更高效的 RPC |
| 切分策略 | 手动实现 | 自动并行化 |
| 多节点 | 不支持 | 支持 |

---

## 8. 小结

nano-vllm 的张量并行实现：

1. **进程管理**：使用 spawn 模式创建多进程
2. **权重切分**：Column Parallel 和 Row Parallel 组合
3. **通信机制**：通过共享内存传递请求，NCCL 进行梯度汇总
4. **模型实现**：QKV 并行、FFN 并行、输出投影

张量并行是扩展大模型推理能力的关键技术，配合 Continuous Batching 和 PagedAttention，nano-vllm 实现了高效的多 GPU 推理服务。

---

## 附录：启动多 GPU 推理

```python
from nanovllm import LLM

# 启动 4 卡推理
llm = LLM(
    model="/path/to/model",
    tensor_parallel_size=4,  # 使用 4 个 GPU
    gpu_memory_utilization=0.9,
)

# 使用方式与单卡相同
outputs = llm.generate(["Hello, world!"], sampling_params)
```

系统会自动将模型权重分布到 4 个 GPU 上，并通过 NCCL 协调计算。