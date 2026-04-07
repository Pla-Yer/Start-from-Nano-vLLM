# 调度器设计：Prefill/Decode 分离与 Chunked Prefill

## 1. 调度器核心职责

调度器(Scheduler)是 vLLM 的核心组件，负责：
1. **请求队列管理**：维护 waiting（等待）和 running（运行）两个队列
2. **资源分配**：决定哪些序列可以运行，分配 KV Cache 块
3. **Preemption**：内存不足时抢占低优先级序列
4. **阶段判定**：区分 prefill（预填充）和 decode（解码）阶段
5. **Chunked Prefill**：将长 prompt 拆成多个 chunk，按 step 与 decode 交替

---

## 2. 队列管理

### 2.1 队列状态

nano-vllm 使用双队列设计：

```python
class Scheduler:
    def __init__(self, config):
        self.waiting: deque[Sequence] = deque()  # 新请求/被抢占的请求
        self.running: deque[Sequence] = deque()  # 正在处理的请求
```

**Sequence 状态机**：
```
WAITING ──(分配资源)──► RUNNING ──(完成)──► FINISHED
   ▲                        │
   └──────(抢占)────────────┘
```

### 2.2 资源限制

两个核心限制条件：

1. **`max_num_seqs`**：最大并发序列数
2. **`max_num_batched_tokens`**：单批次最大 token 数

```python
# 调度循环
while self.waiting and num_seqs < self.max_num_seqs:
    if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
        break  # 达到 token 限制，停止添加
```

---

## 3. Prefill/Decode 分离调度

### 3.1 为什么需要分离？

| 阶段 | 特点 | 优化目标 |
|------|------|----------|
| **Prefill** | 处理整个 prompt，计算量大 | 吞吐量 |
| **Decode** | 每次只生成 1 个 token，计算量小 | 延迟 |

**关键洞察**：
- Prefill 阶段是计算密集型，适合大批量
- Decode 阶段是内存密集型，小批量更高效
- 混合调度可能导致互相阻塞

### 3.1.1 为什么要引入 Chunked Prefill？

传统实现里，只要 `waiting` 队列非空，调度器就会优先把整条 prompt 一次性做完 prefill。这种策略很直接，但在长 prompt 和短 prompt 混合的场景里会暴露两个问题：

- 长 prompt 会持续占满 prefill token budget
- decode 请求即使只差一个 step，也可能被长 prefill 阻塞很久

nano-vllm 现在支持一个兼容当前执行栈的 chunked prefill 版本：

- 每次 prefill 只处理 `prefill_chunk_size` 个 prompt token
- 当 `waiting` 和 `running` 同时非空时，在 step 粒度对 `prefill chunk` 和 `decode` 交替调度

```text
prefill(chunk 1) -> decode -> prefill(chunk 2) -> decode -> ...
```

这样不需要在一个 CUDA batch 内真正混合 prefill 和 decode，也能先解决 decode starvation。

### 3.2 调度算法

nano-vllm 的调度逻辑（`scheduler.py`）可以概括为：

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    if enable_chunked_prefill and waiting and running:
        return alternate_prefill_and_decode()

    scheduled = schedule_prefill()
    if scheduled:
        return scheduled, True

    scheduled = schedule_decode()
    return scheduled, False
```

其中 `schedule_prefill()` 的核心变化是：调度器不再默认吞掉整条 prompt，而是给每个被选中的序列打上一个 `prefill_chunk_size`，`ModelRunner` 只消费这一段 prompt。

```python
token_budget = prefill_chunk_size if enable_chunked_prefill else max_num_batched_tokens

while waiting and num_batched_tokens < token_budget:
    seq = waiting[0]
    if not seq.block_table:
        block_manager.allocate(seq)

    chunk_size = min(seq.num_prompt_tokens_remaining, token_budget - num_batched_tokens)
    seq.prefill_chunk_size = chunk_size
    scheduled.append(seq)

    if chunk_size == seq.num_prompt_tokens_remaining:
        waiting.popleft()
        running.append(seq)
```

### 3.3 调度流程图

```
                    ┌─────────────────────────────────┐
                    │         scheduler.schedule()    │
                    └─────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │ waiting 与 running 同时非空？  │
                    └────────────────┬────────────────┘
                              Yes   │   No
                    ┌───────────────┘   └───────────────┐
                    ▼                                   ▼
            ┌────────────────┐                 ┌────────────────┐
            │ 交替执行       │                 │ 常规流程       │
            │ prefill/decode │                 │ prefill 优先   │
            └───────┬────────┘                 └──────┬─────────┘
                    │                                  │
            ┌───────┴────────┐                 ┌───────┴────────┐
            ▼                ▼                 ▼                ▼
      prefill chunk      decode step        prefill          decode
```

---

## 4. 资源分配与 Preemption

### 4.1 块分配 (Block Allocation)

```python
def allocate(self, seq: Sequence):
    # 为序列分配 KV Cache 块
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)

        # 计算块哈希（用于 prefix caching）
        h = self.compute_hash(token_ids, prefix_hash)

        # 检查缓存
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1:
            # 缓存未命中，分配新块
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
        else:
            # 缓存命中，复用块，增加引用计数
            block.ref_count += 1
            seq.num_cached_tokens += self.block_size

        seq.block_table.append(block_id)
```

### 4.2 Preemption (抢占)

当 decode 阶段发现内存不足时，触发抢占：

```python
def preempt(self, seq: Sequence):
    # 释放序列的 KV Cache 块
    seq.status = SequenceStatus.WAITING
    seq.prefill_chunk_size = 0
    self.block_manager.deallocate(seq)

    # 放回 waiting 队列队首
    self.waiting.appendleft(seq)
```

**抢占策略**：LIFO（后进先出），抢占最新进入 running 队列的序列，因为它们的 prompt 较短，成本较低。

---

## 5. 后处理 (Postprocess)

每次推理后需要更新序列状态。普通 decode 与 chunked prefill 的后处理不同：

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
    for seq, token_id in zip(seqs, token_ids):
        if is_prefill:
            seq.num_cached_tokens += seq.prefill_chunk_size
            if not seq.is_prefill_finished:
                continue

        # 只有最后一个 prefill chunk 或 decode step 才追加新 token
        seq.append_token(token_id)

        # 判断是否完成
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            # 标记完成，释放资源
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

也就是说：

- 中间 prefill chunk 只负责把 KV 写入缓存，不会向用户产出 token
- 只有最后一个 prefill chunk 才会生成首个 output token

`BlockManager` 也做了一个配套修改：对于 cache miss 的 block，不会在 `allocate()` 时立刻注册到 prefix cache，而是等该 block 真正完成 prefill 后，再登记哈希。否则其他请求可能错误复用一个“已经分配但 KV 还没写完”的 block。

**完成条件**：
1. 遇到 EOS token 且未设置 `ignore_eos`
2. 生成的 token 数达到上限 `max_tokens`

---

## 6. Chunked Prefill 配置

```python
llm = LLM(
    model=model_path,
    enable_chunked_prefill=True,
    prefill_chunk_size=2048,
)
```

| 参数 | 作用 |
|------|------|
| `enable_chunked_prefill` | 是否启用 chunked prefill |
| `prefill_chunk_size` | 单次 prefill step 最多处理多少 prompt token |

默认情况下：

- `enable_chunked_prefill=False`
- `prefill_chunk_size` 会回退到 `max_num_batched_tokens`

---

## 7. 性能影响因素

### 6.1 调度参数调优

| 参数 | 影响 |
|------|------|
| `max_num_batched_tokens` | 越大 → prefill 吞吐量越高，但延迟增加 |
| `max_num_seqs` | 越大 → 并发度越高，但内存竞争加剧 |

### 6.2 调度策略对比

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **FCFS** | 先来先服务 | 公平 | 可能不最优 |
| **Preemptive** | 可抢占 | 内存利用率高 | 实现复杂 |
| **Priority** | 优先级队列 | 关键请求优先 | 可能饿死 |

nano-vllm 采用近似 FCFS + Preemptive 策略。

---

## 8. 小结

调度器的核心设计：
1. **双队列**：waiting（等待）和 running（运行）
2. **Prefill 优先**：默认先处理 waiting 队列的 prefill 请求
3. **资源限制**：通过 `max_num_batched_tokens` 和 `max_num_seqs` 控制
4. **Preemption**：内存不足时抢占低优先级序列
5. **Decode 处理**：running 队列中的序列执行 decode
6. **Chunked Prefill**：长 prompt 可按 chunk 切分，并与 decode 在 step 粒度交替

理解调度器是理解 vLLM 性能优化的关键，下一篇将深入 KV Cache 管理。
