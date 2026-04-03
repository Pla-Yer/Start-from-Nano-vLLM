# 调度器设计：Prefill/Decode 分离策略

## 1. 调度器核心职责

调度器(Scheduler)是 vLLM 的核心组件，负责：
1. **请求队列管理**：维护 waiting（等待）和 running（运行）两个队列
2. **资源分配**：决定哪些序列可以运行，分配 KV Cache 块
3. **Preemption**：内存不足时抢占低优先级序列
4. **阶段判定**：区分 prefill（预填充）和 decode（解码）阶段

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

### 3.2 调度算法

nano-vllm 的调度逻辑（`scheduler.py`）：

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # ===== 第一阶段：Prefill =====
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0

    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]  # 取队首

        # 检查资源是否足够
        if (num_batched_tokens + len(seq) > self.max_num_batched_tokens or
            not self.block_manager.can_allocate(seq)):
            break  # 资源不足，停止 prefill

        # 分配资源
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens

        # 状态变更
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)

    # 如果有 prefill 请求，返回 prefill
    if scheduled_seqs:
        return scheduled_seqs, True  # is_prefill=True

    # ===== 第二阶段：Decode =====
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()

        # 检查是否可以追加（内存是否足够）
        while not self.block_manager.can_append(seq):
            if self.running:
                # 抢占 running 队列最后一个序列
                self.preempt(self.running.pop())
            else:
                # 没有其他序列可抢占，抢占当前序列
                self.preempt(seq)
                break
        else:
            # 成功分配内存
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)

    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False  # is_prefill=False
```

### 3.3 调度流程图

```
                    ┌─────────────────────────────────┐
                    │         scheduler.schedule()    │
                    └─────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │     waiting 队列有请求？        │
                    └────────────────┬────────────────┘
                              Yes   │   No
                    ┌───────────────┐ │
                    ▼               ▼ ▼
               ┌────────┐      ┌──────────────┐
               │ Prefill│      │ Decode 阶段  │
               │ 阶段   │      │ running 队列 │
               └───┬────┘      └──────┬───────┘
                   │                  │
        ┌──────────┼──────────┐       │
        ▼          ▼          ▼       ▼
    资源足够？   达到 max   达到 max  内存不足？
        │        seqs？    tokens？   │
        ▼          ▼          ▼       ▼
     分配块    停止 prefill 停止      Preemption
                                  (抢占其他序列)
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
    self.block_manager.deallocate(seq)

    # 放回 waiting 队列队首
    self.waiting.appendleft(seq)
```

**抢占策略**：LIFO（后进先出），抢占最新进入 running 队列的序列，因为它们的 prompt 较短，成本较低。

---

## 5. 后处理 (Postprocess)

每次推理后需要更新序列状态：

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
    for seq, token_id in zip(seqs, token_ids):
        # 添加生成的 token
        seq.append_token(token_id)

        # 判断是否完成
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            # 标记完成，释放资源
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

**完成条件**：
1. 遇到 EOS token 且未设置 `ignore_eos`
2. 生成的 token 数达到上限 `max_tokens`

---

## 6. 性能影响因素

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

## 7. 小结

调度器的核心设计：
1. **双队列**：waiting（等待）和 running（运行）
2. **Prefill 优先**：先处理 waiting 队列的 prefill 请求
3. **资源限制**：通过 `max_num_batched_tokens` 和 `max_num_seqs` 控制
4. **Preemption**：内存不足时抢占低优先级序列
5. **Decode 处理**：running 队列中的序列执行 decode

理解调度器是理解 vLLM 性能优化的关键，下一篇将深入 KV Cache 管理。