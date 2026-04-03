# KV Cache 管理：PagedAttention 与 Block Manager

## 1. PagedAttention 核心思想

### 1.1 传统方式的局限性

传统方法将每个序列的 KV Cache 存储为连续内存块：

```
Sequence 1: [KV KV KV KV KV KV KV KV]  (8 tokens)
Sequence 2: [KV KV KV]                  (3 tokens)
Sequence 3: [KV KV KV KV KV]            (5 tokens)
```

**问题**：
- 序列长度动态变化，需要预分配最大长度
- 内存碎片：释放后的空隙无法利用
- 内存浪费：实际使用远小于预分配

### 1.2 PagedAttention 解决方案

类比操作系统虚拟内存的页表机制：

```
Sequence 1: [Page 0] [Page 1] [Page 2]
Sequence 2: [Page 3] (共享 Page 0) [Page 4]
Sequence 3: [Page 0] [Page 5]
            ↑
         共享前缀
```

**优势**：
- 按需分配页，无需预分配最大长度
- 页可以共享（如系统提示）
- 释放时只释放实际使用的页
- 支持高效的 prefix caching

---

## 2. Block Manager 实现

### 2.1 核心数据结构

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id        # 块 ID
        self.ref_count = 0              # 引用计数（共享时 >1）
        self.hash = -1                  # 块内容的哈希（用于缓存）
        self.token_ids = []             # 存储的 token 列表

class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size    # 默认 256 tokens
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}  # 哈希 -> 块 ID
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲块
        self.used_block_ids: set[int] = set()  # 已使用块
```

### 2.2 块分配 (Allocation)

```python
def allocate(self, seq: Sequence):
    """为序列分配 KV Cache 块"""
    assert not seq.block_table  # 确保序列没有已分配的块

    for i in range(seq.num_blocks):
        token_ids = seq.block(i)  # 获取第 i 个块的内容

        # 计算块的哈希值（用于缓存匹配）
        if len(token_ids) == self.block_size:
            h = self.compute_hash(token_ids, prefix_hash)
        else:
            h = -1  # 不满的块不计算哈希

        # 查找缓存
        block_id = self.hash_to_block_id.get(h, -1)

        # 检查缓存是否有效
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        else:
            cache_miss = False

        if cache_miss:
            # 缓存未命中，分配新块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # 缓存命中，复用块
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                # 块正在被其他序列使用，增加引用计数
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                # 块空闲，分配给当前序列
                block = self._allocate_block(block_id)

        # 更新哈希映射
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id

        seq.block_table.append(block_id)
```

### 2.3 块释放 (Deallocation)

```python
def deallocate(self, seq: Sequence):
    """释放序列的 KV Cache 块"""
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1

        # 引用计数为 0 时，标记为空闲
        if block.ref_count == 0:
            self._deallocate_block(block_id)

    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

---

## 3. Prefix Caching

### 3.1 缓存原理

通过哈希匹配检测相同的前缀：

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    """计算块的哈希值"""
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 包含前一块哈希
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

### 3.2 缓存命中场景

1. **系统提示共享**：多个请求使用相同的 system prompt
2. **few-shot 示例**：相同的 few-shot 示例
3. **重复前缀**：相同的前缀内容

### 3.3 缓存命中时的处理

```python
# 在 model_runner.py 的 prepare_prefill 中
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 有前缀缓存
    block_tables = self.prepare_block_tables(seqs)
```

- 使用 `flash_attn_varlen_func` 的 `block_table` 参数
- 只计算新 token 的注意力，前缀部分从缓存读取

---

## 4. 内存计算与分配

### 4.1 KV Cache 内存估算

```python
def allocate_kv_cache(self):
    """根据 GPU 显存计算可分配的块数"""
    free, total = torch.cuda.mem_get_info()
    used = total - free

    # 获取峰值内存和当前使用
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    # 每个块的字节数
    num_kv_heads = hf_config.num_key_value_heads // world_size
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    block_bytes = (2 * num_layers * block_size *
                   num_kv_heads * head_dim *
                   dtype.itemsize)

    # 可用块数
    num_kvcache_blocks = int(
        (total * gpu_memory_utilization - used - peak + current) // block_bytes
    )
```

### 4.2 内存分配公式

```
KV Cache 内存 = 2 × num_layers × block_size × num_kv_heads × head_dim × dtype_bytes
```

以 Qwen2-0.5B 为例：
- num_layers = 24
- block_size = 256
- num_kv_heads = 2
- head_dim = 64
- dtype = fp16 (2 bytes)

每块 = 2 × 24 × 256 × 2 × 64 × 2 = 3.1 MB

---

## 5. 块追加与动态扩展

### 5.1 Decode 阶段追加块

```python
def can_append(self, seq: Sequence) -> bool:
    """检查是否可以追加新 token"""
    # 需要新块的条件：当前块已满
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    """为序列追加新的块或更新哈希"""
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]

    if len(seq) % self.block_size == 1:
        # 需要分配新块
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)

    elif len(seq) % self.block_size == 0:
        # 块已满，计算哈希用于缓存
        token_ids = seq.block(seq.num_blocks - 1)
        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
```

### 5.2 追加流程

```
Decode 阶段 token 1: [Block 0] 有 1 个 token
Decode 阶段 token 2: [Block 0] 有 2 个 token
...
Decode 阶段 token 256: [Block 0] 有 256 个 token (满)，计算哈希
Decode 阶段 token 257: [Block 0] 满，[Block 1] 新分配
```

---

## 6. 缓存命中率统计

### 6.1 追踪指标

在 `allocate` 过程中可以追踪：

```python
# 统计信息
cache_hits = 0  # 缓存命中次数
cache_misses = 0  # 缓存未命中次数
total_blocks = 0  # 总分配块数

# 计算命中率
hit_rate = cache_hits / total_blocks if total_blocks > 0 else 0
```

### 6.2 影响因素

| 因素 | 对命中率的影响 |
|------|----------------|
| 相同 system prompt | 高 |
| few-shot 示例相同 | 高 |
| block_size 越大 | 命中率越低（粒度粗） |
| 请求多样性高 | 命中率低 |

---

## 7. 与 vLLM 的对比

| 特性 | nano-vllm | vLLM |
|------|------------|------|
| 块大小 | 固定 256 | 可配置 |
| 块分配 | 简单分配 | 混合块大小 |
| 缓存策略 | 哈希匹配 | 更复杂的缓存淘汰 |
| 共享块 | 引用计数 | 块引用追踪 |

---

## 8. 小结

Block Manager 的核心设计：

1. **分页管理**：将 KV Cache 分为固定大小的块
2. **哈希缓存**：通过哈希匹配实现 prefix caching
3. **引用计数**：支持块共享和正确释放
4. **动态扩展**：decode 阶段按需分配新块
5. **内存估算**：根据 GPU 显存动态计算可用块数

理解 KV Cache 管理是优化推理性能的关键，下一篇将介绍模型推理与 CUDA Graph。