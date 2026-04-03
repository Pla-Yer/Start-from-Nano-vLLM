# vLLM 整体架构与请求流程

## 1. vLLM 核心设计思想

### 1.1 传统推理引擎的瓶颈

大语言模型(LLM)推理面临两个核心挑战：

1. **KV Cache 内存碎片**：传统方式将序列的 KV Cache 存储为连续内存，序列长度变化时导致内存碎片和浪费
2. **Batch 处理效率低**：静态 batching 无法动态适应不同长度请求，GPU 利用率低

### 1.2 vLLM 的创新解决方案

**PagedAttention**：将 KV Cache 分页管理，类比操作系统虚拟内存的页表机制
- 序列的 KV Cache 不需要连续存储
- 按固定块大小(默认 256 tokens)分页
- 支持内存共享和高效释放

**Continuous Batching**：动态批处理
- 不等待整个 batch 完成就处理新请求
- 最大化 GPU 利用率
- 显著提升吞吐量

---

## 2. nano-vllm 整体架构

### 2.1 模块分层

```
┌─────────────────────────────────────────┐
│          nanovllm/llm.py                │
│            LLM 类 (入口)                 │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│       nanovllm/engine/llm_engine.py     │
│           引擎层 (编排)                  │
└─────────────────────────────────────────┘
         │              │              │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │Scheduler│    │Model    │    │多进程   │
    │         │    │Runner   │    │管理     │
    └─────────┘    └─────────┘    └─────────┘
         │              │
    ┌────▼────┐    ┌────▼────┐
    │Sequence │    │Block    │
    │         │    │Manager  │
    └─────────┘    └─────────┘
         │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │Attention│    │Sampler  │    │Linear   │
    │         │    │         │    │(TP)     │
    └─────────┘    └─────────┘    └─────────┘
```

### 2.2 核心文件说明

| 文件 | 职责 |
|------|------|
| `llm.py` | LLM 类，继承 LLMEngine |
| `llm_engine.py` | 推理流程编排、多进程管理、请求入口 |
| `scheduler.py` | 请求调度，prefill/decode 分离 |
| `model_runner.py` | 模型执行，CUDA Graph，KV Cache 管理 |
| `block_manager.py` | KV Cache 块分配，Prefix Caching |
| `sequence.py` | 单个请求的状态管理 |
| `attention.py` | FlashAttention，KV Cache 写回 |
| `sampler.py` | 采样策略，torch.compile |
| `linear.py` | 张量并行线性层 |

---

## 3. 请求处理流程

### 3.1 请求生命周期

```
用户请求 ──► 添加到 waiting 队列 ──► Scheduler 调度
                                              │
                                              ▼
                                        ┌────────────┐
                                        │ 预填充阶段 │
                                        │ (Prefill)  │
                                        └─────┬──────┘
                                              │
                                              ▼
                                        ┌────────────┐
                                        │ 解码阶段   │
                                        │ (Decode)   │
                                        └─────┬──────┘
                                              │
                                              ▼
                                         输出完成
```

### 3.2 详细流程 (基于 llm_engine.py)

```python
# 1. 添加请求
def add_request(self, prompt, sampling_params):
    # 将字符串转为 token IDs
    prompt = self.tokenizer.encode(prompt)
    # 创建 Sequence 对象
    seq = Sequence(prompt, sampling_params)
    # 加入 waiting 队列
    self.scheduler.add(seq)

# 2. 推理循环
def generate(self, prompts, sampling_params):
    # 批量添加请求
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)

    outputs = {}
    while not self.is_finished():
        # 调度：选择要处理的序列
        seqs, is_prefill = self.scheduler.schedule()
        # 执行：运行模型
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理：更新序列状态
        self.scheduler.postprocess(seqs, token_ids)
        # 收集完成的输出
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids

    return [tokenizer.decode(ids) for ids in outputs]
```

### 3.3 调度策略 (scheduler.py)

```python
def schedule(self):
    # 优先处理 prefill（waiting 队列）
    scheduled_seqs = []
    while self.waiting and num_seqs < max_num_seqs:
        # 检查 token 数和块分配
        if can_allocate(seq):
            allocate_block(seq)
            scheduled_seqs.append(seq)  # 标记为 prefill

    if scheduled_seqs:
        return scheduled_seqs, True  # True = prefill

    # prefill 完成后，处理 decode（running 队列）
    while self.running and num_seqs < max_num_seqs:
        if can_append(seq):
            may_append(seq)
            scheduled_seqs.append(seq)  # 标记为 decode

    return scheduled_seqs, False  # False = decode
```

**关键策略**：
- **Prefill 优先**：先处理等待中的 prompt，最大化首 token 响应速度(TTFT)
- **资源限制**：`max_num_batched_tokens` 和 `max_num_seqs` 控制单轮处理量
- **Preemption**：内存不足时抢占 running 序列，释放 KV Cache

---

## 4. 关键配置参数

```python
@dataclass
class Config:
    model: str                              # 模型路径
    max_num_batched_tokens: int = 16384     # 单批次最大 token 数
    max_num_seqs: int = 512                 # 最大并发序列数
    max_model_len: int = 4096               # 单序列最大长度
    gpu_memory_utilization: float = 0.9     # GPU 显存用于 KV Cache 比例
    tensor_parallel_size: int = 1           # 张量并行 GPU 数量
    enforce_eager: bool = False             # 禁用 CUDA Graph
    kvcache_block_size: int = 256           # KV Cache 块大小
```

---

## 5. 核心设计模式

### 5.1 流水线执行

```
Scheduler.add() ──► ModelRunner.run() ──► Scheduler.postprocess()
     │                     │                      │
  添加请求              计算 logits            更新状态
```

### 5.2 多进程架构

- **主进程 (rank 0)**：调度、采样、结果汇总
- **工作进程 (rank 1,2,...)**：模型前向传播
- **通信**：通过 `multiprocessing.shared_memory` 传递请求

### 5.3 CUDA Graph 优化

- **Prefill 阶段**：直接执行（变化大，无法缓存）
- **Decode 阶段**：使用 CUDA Graph 加速（批量小，模式固定）
- **条件**：`enforce_eager=False` 且 batch_size ≤ 512

---

## 6. 性能指标

在 `generate()` 中追踪：

```python
prefill_throughput = num_tokens / (perf_counter() - t)  # tokens/s
decode_throughput = -num_tokens / (perf_counter() - t)  # tokens/s
```

- **Prefill 吞吐量**：处理 prompt 的速度
- **Decode 吞吐量**：生成 token 的速度

---

## 7. 小结

vLLM/nano-vllm 的核心创新：
1. **PagedAttention**：将 KV Cache 分页管理，消除内存碎片
2. **Continuous Batching**：动态批处理，最大化 GPU 利用率
3. **调度策略**：prefill/decode 分离，资源智能分配

理解这些核心概念，为后续深入各模块打下基础。