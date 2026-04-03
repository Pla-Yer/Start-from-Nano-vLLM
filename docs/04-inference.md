# 模型推理与 CUDA Graph

## 1. 推理流程概述

### 1.1 ModelRunner 职责

`ModelRunner` 是 nano-vllm 中负责模型执行的核心组件，主要功能：
1. **模型加载**：从 HuggingFace 格式加载模型
2. **KV Cache 分配**：预分配 GPU 显存作为 KV Cache
3. **输入准备**：将序列数据转换为模型输入
4. **模型执行**：运行前向传播
5. **CUDA Graph 捕获**：优化 decode 阶段性能

### 1.2 初始化流程

```python
def __init__(self, config: Config, rank: int, event: Event | list[Event]):
    # 1. 初始化分布式
    dist.init_process_group("nccl", ..., world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    # 2. 加载模型
    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model)

    # 3. 初始化采样器
    self.sampler = Sampler()

    # 4. 模型预热
    self.warmup_model()

    # 5. 分配 KV Cache
    self.allocate_kv_cache()

    # 6. 捕获 CUDA Graph（如果启用）
    if not self.enforce_eager:
        self.capture_cudagraph()
```

---

## 2. 输入准备

### 2.1 Prefill 阶段输入准备

```python
def prepare_prefill(self, seqs: list[Sequence]):
    """准备 prefill 阶段的输入"""
    input_ids = []
    positions = []
    cu_seqlens_q = [0]  # query 的 cumulative sequence lengths
    cu_seqlens_k = [0]  # key/value 的 cumulative sequence lengths
    slot_mapping = []
    block_tables = None

    for seq in seqs:
        seqlen = len(seq)
        # 只取未缓存的部分（从 num_cached_tokens 开始）
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(range(seq.num_cached_tokens, seqlen))

        seqlen_q = seqlen - seq.num_cached_tokens
        seqlen_k = seqlen
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

        # 构建 slot_mapping（KV Cache 位置映射）
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * self.block_size
            if i != seq.num_blocks - 1:
                end = start + self.block_size
            else:
                end = start + seq.last_block_num_tokens
            slot_mapping.extend(range(start, end))

    # 如果有前缀缓存，传入 block_tables
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)

    # 转换为 Tensor 并移到 GPU
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    # 设置全局上下文
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                slot_mapping, None, block_tables)

    return input_ids, positions
```

### 2.2 Decode 阶段输入准备

```python
def prepare_decode(self, seqs: list[Sequence]):
    """准备 decode 阶段的输入"""
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []

    for seq in seqs:
        # decode 只需要最后一个 token
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        # 计算当前 token 在 KV Cache 中的位置
        slot_mapping.append(
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        )

    # 转换为 Tensor
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    block_tables = self.prepare_block_tables(seqs)

    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)

    return input_ids, positions
```

### 2.3 Prefill vs Decode 输入对比

| 特性 | Prefill | Decode |
|------|---------|--------|
| input_ids | 多个 token | 单个 token (last_token) |
| positions | 连续 | 单一位置 |
| cu_seqlens | 变长序列 | 所有序列长度=1 |
| slot_mapping | 多个位置 | 单个位置 |
| block_tables | 可选（prefix cache 时需要） | 必须 |

---

## 3. 模型执行

### 3.1 运行入口

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    # 1. 根据阶段准备输入
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)

    # 2. 准备采样参数（只在 rank 0）
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

    # 3. 运行模型
    logits = self.run_model(input_ids, positions, is_prefill)

    # 4. 采样（只在 rank 0）
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

    # 5. 清理上下文
    reset_context()

    return token_ids
```

### 3.2 模型前向

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    # 条件判断：何时使用 CUDA Graph
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # Prefill 阶段 或 禁用 CUDA Graph 或 batch 太大：直接执行
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # Decode 阶段：使用 CUDA Graph 加速
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]

        # 更新 graph 中的输入
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        # 重放 CUDA Graph
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

---

## 4. CUDA Graph 详解

### 4.1 什么是 CUDA Graph？

CUDA Graph 是 NVIDIA 推出的优化技术，可以：
1. **捕获**整个计算图（包括 kernel 启动）
2. **一次性提交**整个图，减少 CPU-GPU 通信开销
3. **重放**时无需重复 kernel 编排

### 4.2 为什么 Decode 阶段适合 CUDA Graph？

| 阶段 | 计算模式 | 适合 CUDA Graph？ |
|------|----------|-------------------|
| Prefill | 变化大（不同 prompt 不同长度） | 否 |
| Decode | 固定（每次只生成 1 个 token） | 是 |

Decode 阶段的计算模式高度规律：
- 输入形状相对固定
- KV Cache 访问模式固定
- 可以预先捕获并重放

### 4.3 CUDA Graph 捕获

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config

    # 最大 batch size 和 block 数
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

    # 预分配 tensors
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    # 定义不同 batch size 的 graph
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    # 从大到小捕获（共享内存池）
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()

        # 设置上下文
        set_context(False,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs])

        # warmup
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 捕获
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 记录内存池
        if self.graph_pool is None:
            self.graph_pool = graph.pool()

        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    # 保存可变 tensors
    self.graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
```

### 4.4 内存池共享

```
graph_bs = [512, 496, 480, ..., 16, 8, 4, 2, 1]

捕获顺序：512 → 496 → ... → 1
         ↓     ↓        ↓
      graph graph ...  graph
         ↓     ↓        ↓
      共享 memory pool（从大到小共享）
```

**关键优化**：从大到小捕获，小的 graph 可以使用大 graph 分配的内存池。

---

## 5. 采样器 (Sampler)

### 5.1 采样实现

```python
class Sampler(nn.Module):
    @torch.compile  # 使用 torch.compile 优化
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 1. 应用温度
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 2. 计算概率
        probs = torch.softmax(logits, dim=-1)

        # 3. 贪婪/随机采样
        # 使用指数采样实现温度采样
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        return sample_tokens
```

**torch.compile 优化**：将采样逻辑编译为优化的 CUDA kernel。

### 5.2 采样策略

| 参数 | 说明 |
|------|------|
| `temperature` | 温度参数，>1 增加随机性，<1 增加确定性，=0 贪婪 |
| `max_tokens` | 最大生成 token 数 |
| `ignore_eos` | 是否忽略 EOS token |

---

## 6. 性能优化技巧

### 6.1 Pin Memory + Async Copy

```python
# 使用 pin_memory 加速 CPU → GPU 传输
input_ids = torch.tensor(..., pin_memory=True).cuda(non_blocking=True)
```

### 6.2 inference_mode vs no_grad

```python
@torch.inference_mode()  # 比 no_grad 更快
def run_model(self, ...):
    ...
```

### 6.3 条件跳过 CUDA Graph

```python
# 以下情况不使用 CUDA Graph
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
```

**原因**：
- Prefill：计算模式不固定
- enforce_eager：用户禁用
- batch > 512：超出捕获范围

---

## 7. 多进程执行

### 7.1 进程角色

- **Rank 0 (主进程)**：调度、采样、结果汇总
- **Rank 1,2,... (工作进程)**：模型前向传播

### 7.2 通信机制

```python
# 写入请求（Rank 0）
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()

# 读取请求（其他 Rank）
def read_shm(self):
    self.event.wait()  # 等待主进程信号
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args
```

---

## 8. 小结

ModelRunner 的核心设计：

1. **输入准备**：根据 prefill/decode 阶段准备不同格式的输入
2. **模型执行**：调用模型前向传播
3. **CUDA Graph**：为 decode 阶段捕获和重放计算图
4. **采样**：使用 torch.compile 优化的采样器
5. **多进程**：通过共享内存协调多 GPU 推理

CUDA Graph 是提升 decode 吞吐量的关键技术，下一篇将介绍分布式推理（Tensor Parallelism）。