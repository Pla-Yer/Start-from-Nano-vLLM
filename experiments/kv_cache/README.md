# KV Cache 实验

本目录包含针对 KV Cache 性能优化的基准测试实验,通过实际测试数据深入理解 KV Cache 的工作原理和优化策略。

## 实验一: Block Size 基准测试 (`benchmark.py`)

### 实验目的
理解 KV Cache 的 block size 参数如何影响内存管理和推理性能。

### 核心概念
**Block Size** 是 KV Cache 内存管理的最小分配单元:
- KV Cache 被划分为固定大小的 block
- 每个 block 存储一定数量的 token 的 KV 值
- Block size 决定了内存分配的粒度

### 测试参数
```python
block_sizes = [256, 512, 1024]  # 测试的三种block size
max_tokens = 64                  # 每个请求最多生成64个token
num_repeats = 4                  # 8个基础提示 × 4次重复 = 32个请求
```

### 实验结果 (Qwen3-0.6B)
```
Block Size   Time(s)   Tok/s      Req/s      Avg Latency(s)
256          0.245     8359.77    130.62     0.008
512          0.256     8003.96    125.06     0.008
1024         0.250     8180.82    127.83     0.008
```

### 结果分析

**关键发现: Block size 对性能影响很小**

1. **性能差异微乎其微**
   - 最大差异仅 4.4% (256 vs 512)
   - 三种配置的吞吐量都在 8000-8400 tok/s 范围内
   - 平均延迟稳定在 8ms 左右

2. **为什么影响这么小?**
   - **测试场景特点**: 每个请求生成 64 个 token,相对较短
   - **内存分配次数**: 即使最小的 block_size=256,对于 64 token 的请求也只需 1 个 block
   - **无内存压力**: 32 个请求的总内存需求远小于 GPU 内存容量

3. **Block size 的实际影响场景**
   - **长序列生成**: 当 max_tokens >> block_size 时,需要多次分配 block
   - **高并发场景**: 大量并发请求时,block 分配和回收的开销会累积
   - **内存碎片**: 较大的 block size 可能减少碎片,但增加单次分配开销

4. **参数选择建议**
   - **256**: 适合短序列、高并发场景,内存利用率高
   - **512**: 平衡选择,适合大多数场景
   - **1024**: 适合长序列场景,减少分配次数

### 教学要点
- Block size 是内存管理粒度参数,不是性能调优的"银弹"
- 在内存充足、序列较短的场景下,block size 影响有限
- 真正的性能差异出现在内存压力大的场景

---

## 实验二: Prefix Cache 基准测试 (`prefix.py`)

### 实验目的
验证前缀缓存(Prefix Cache)优化机制的有效性,理解其工作原理。

### 核心概念
**Prefix Cache** 利用请求间的共同前缀来减少重复计算:
- 多个请求共享相同的系统提示或上下文
- 只需计算一次共享前缀的 KV Cache
- 后续请求直接复用缓存的 KV 值

### 测试参数
```python
prefix_repeat_tokens = 100  # 前缀包含约100个token
max_tokens = 8              # 每个请求生成8个token(短输出)
num_repeats = 4             # 8个基础提示 × 4次重复 = 32个请求
```

### 测试场景设计
1. **Shared Prefix (共享前缀)**
   ```python
   # 所有请求共享同一个前缀
   prefix = "system: You are a careful assistant. policy0 policy1 ... policy99"
   prompts = [prefix + question for question in test_questions]
   ```

2. **Unique Prefix (唯一前缀)**
   ```python
   # 每个请求的前缀内容不同
   for i in range(num_requests):
       prefix_i = f"system: ... policy{i}_0 policy{i}_1 ... policy{i}_99"
       prompts.append(prefix_i + question_i)
   ```

### 实验结果 (Qwen3-0.6B)
```
With shared prefix:
  Time: 0.075s
  Throughput (tok/s): 3415.71
  Throughput (req/s): 426.96

Without shared prefix:
  Time: 0.362s
  Throughput (tok/s): 707.40
  Throughput (req/s): 88.42

Improvement:
  Time reduction: 79.3%
  Throughput increase: 382.9%
```

### 结果分析

**关键发现: 前缀缓存带来巨大性能提升**

1. **性能提升显著**
   - 时间减少 79.3% (0.362s → 0.075s)
   - 吞吐量提升 382.9% (707 → 3416 tok/s)
   - 请求吞吐量提升 4.8 倍 (88 → 427 req/s)

2. **为什么提升这么大?**
   - **前缀长度占比高**: 100 token 前缀 vs 8 token 输出
   - **计算量对比**:
     - Unique: 每个请求计算 100 + 8 = 108 token 的 KV
     - Shared: 第一个请求计算 100 token,后续请求复用,只计算 8 token
   - **理论加速比**: (100 + 8) / 8 = 13.5 倍
   - **实际加速比**: 3415.71 / 707.40 = 4.8 倍(受其他开销影响)

3. **参数对结果的影响**
   - `prefix_repeat_tokens = 100`: 前缀越长,缓存收益越大
   - `max_tokens = 8`: 输出越短,前缀占比越高,收益越明显
   - 如果 `max_tokens = 100`,前缀占比下降,收益会减小

4. **实际应用场景**
   - **对话系统**: 所有请求共享系统提示
   - **RAG 应用**: 共享检索到的文档上下文
   - **Few-shot Learning**: 共享示例前缀

### 教学要点
- 前缀缓存是 vLLM 的核心优化之一
- 性能提升取决于前缀长度与输出长度的比例
- 在实际应用中,系统提示、文档上下文等都是天然的前缀缓存机会

---

## 技术实现细节

### 子进程隔离机制
```python
# 每个测试用例在独立子进程中运行
subprocess.run([sys.executable, script_path, "worker", ...])
```
**原因**: 避免 `torch.distributed` 重复初始化,确保测试公平性

### 预热机制
```python
# 正式测试前运行2个请求预热
_ = llm.generate(prompts[:2], sp, use_tqdm=False)
```
**原因**: 消除首次推理的初始化开销(模型加载、CUDA初始化等)

---

## 实验总结

### Block Size vs Prefix Cache
- **Block Size**: 内存管理参数,在常规场景下影响有限
- **Prefix Cache**: 核心优化机制,能带来数量级的性能提升

### 参数调优建议
1. **Block Size**: 默认值即可,除非遇到内存压力问题
2. **Prefix Cache**: 尽可能利用请求间的共同前缀(系统提示、上下文等)

### 依赖
- `nanovllm`: 核心 vLLM 实现
- `torch`: PyTorch 框架
- Python >= 3.10
