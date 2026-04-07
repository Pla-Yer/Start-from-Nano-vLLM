# Batching 实验

本目录包含针对批处理策略的性能基准测试,通过对比静态批处理(Static Batching)与连续批处理(Continuous Batching),深入理解批处理对推理性能的影响。

## 实验一: 静态批处理 (`static.py`)

### 实验目的
理解固定 batch size 如何影响吞吐量和延迟的权衡关系。

### 核心概念
**静态批处理** 将请求按固定大小分批处理:
- 等待凑够一批请求才开始推理
- 批内所有请求并行处理
- 当前批次完成后才处理下一批

### 测试参数
```python
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]  # 测试的batch size范围
max_tokens = 256                              # 每个请求生成256个token
num_requests = 128                            # 总请求数(8个基础提示 × 16次重复)
temperature = 0.7                             # 采样温度
```

### 实验结果 (Qwen3-0.6B)
```
Batch Size   Requests   Total Time(s)  Avg Latency(s)  Tok/s      Req/s
1            128        107.625        0.841           304.46     1.19
2            128        50.371         0.787           650.53     2.54
4            128        26.067         0.815           1257.08    4.91
8            128        15.368         0.961           2132.17    8.33
16           128        8.384          1.048           3908.61    15.27
32           128        4.976          1.244           6585.45    25.72
64           128        3.473          1.737           9434.26    36.85
128          128        2.508          2.508           13065.22   51.04
```

### 结果分析

**关键发现: Batch size 与性能的非线性关系**

1. **吞吐量随 batch size 单调递增**
   - Batch=1: 304 tok/s
   - Batch=128: 13065 tok/s (提升 43 倍)
   - 说明 GPU 并行计算能力得到充分利用

2. **延迟随 batch size 增加**
   - Batch=1: 0.841s
   - Batch=128: 2.508s (增加 3 倍)
   - 单个请求需要等待整个批次完成

3. **吞吐量增长规律**
   ```
   Batch   Tok/s    增长倍数  理论倍数
   1→2     304→651   2.14x     2x
   2→4     651→1257  1.93x     2x
   4→8     1257→2132 1.70x     2x
   8→16    2132→3909 1.83x     2x
   16→32   3909→6585 1.68x     2x
   32→64   6585→9434 1.43x     2x
   64→128  9434→13065 1.38x    2x
   ```
   
   **观察:**
   - 小 batch 时,增长接近理论值 2x
   - 大 batch 时,增长逐渐放缓(1.38x)
   - 说明存在 GPU 计算资源的瓶颈

4. **为什么大 batch 收益递减?**
   - **GPU 内存带宽瓶颈**: 大 batch 需要更多内存访问
   - **计算资源竞争**: GPU SM 数量有限
   - **KV Cache 开销**: 大 batch 的 KV Cache 管理开销增加

5. **参数对结果的影响**
   - **max_tokens = 256**: 长序列放大了批处理收益
     - 如果 max_tokens = 16,批处理收益会减小
   - **num_requests = 128**: 总请求数足够多,能充分测试各 batch size

### 教学要点
- 批处理是提升吞吐量的核心手段
- 吞吐量与延迟存在权衡关系
- 存在最优 batch size,超过后收益递减

---

## 实验二: 连续批处理 (`continuous.py`)

### 实验目的
对比静态批处理与连续批处理在动态请求场景下的性能差异。

### 核心概念
**连续批处理(Continuous Batching)** 是 vLLM 的核心创新:
- 请求到达后立即开始处理
- 不同请求可以处于不同的生成阶段
- 新请求可以加入正在运行的批次
- 完成的请求立即释放资源,新请求填补空位

### 测试参数
```python
static_batch_sizes = [1, 8, 16, 32]  # 静态批处理的测试配置
arrival_rate = 10.0                   # 平均每秒10个请求到达
jitter = 0.02                         # 到达时间的随机抖动(±0.02秒)
flush_timeout = 0.5                   # 静态批处理的刷新超时
max_tokens = 256                      # 每个请求生成256个token
num_requests = 64                     # 总请求数(8个基础提示 × 8次重复)
```

### 请求到达模型
```python
# 请求按泊松过程到达
arrival_time_i = i / arrival_rate + random.uniform(-jitter, jitter)
# 例如: 第0个请求在0s到达,第1个请求在0.1s到达,第2个请求在0.2s到达...
```

### 实验结果 (Qwen3-0.6B)
```
Strategy        Total Time(s)  Avg Lat(s)  P50 Lat(s)  P95 Lat(s)  Tok/s    Req/s   Avg BS
static_1        53.960         24.211      24.514      45.462      303.64   1.19    1.00
static_8        8.752          1.786       1.811       2.466       1872.09  7.31    7.11
static_16       7.288          1.415       1.408       1.902       2248.17  8.78    9.14
static_32       8.085          1.452       1.460       1.957       2026.52  7.92    8.00
continuous      7.193          1.059       1.076       1.101       2277.92  8.90    0.00

Actual batch size histogram:
static_1: {1: 64}
static_8: {2: 1, 6: 1, 8: 7}
static_16: {6: 1, 8: 1, 10: 5}
static_32: {1: 1, 6: 1, 8: 1, 9: 1, 10: 4}
```

### 结果分析

**关键发现: 连续批处理在延迟和吞吐量上全面优于静态批处理**

1. **延迟对比**
   ```
   策略         Avg Lat   P50 Lat   P95 Lat
   static_1     24.21s    24.51s    45.46s   ← 极差,请求排队时间长
   static_8     1.79s     1.81s     2.47s    ← 较好,但P95仍有波动
   static_16    1.42s     1.41s     1.90s    ← 更好
   static_32    1.45s     1.46s     1.96s    ← 与static_16相近
   continuous   1.06s     1.08s     1.10s    ← 最佳,P95控制最好
   ```
   
   **关键观察:**
   - 连续批处理的 P95 延迟仅 1.10s,远优于所有静态配置
   - static_1 的延迟高达 24s,因为请求需要排队等待
   - static_16 和 static_32 性能相近,说明 batch size 过大反而无益

2. **吞吐量对比**
   ```
   策略         Tok/s    Req/s
   static_1     304      1.19    ← 吞吐量最低
   static_8     1872     7.31    ← 中等
   static_16    2248     8.78    ← 较好
   static_32    2027     7.92    ← 反而下降
   continuous   2278     8.90    ← 最佳
   ```
   
   **关键观察:**
   - 连续批处理吞吐量最优(2278 tok/s)
   - static_32 吞吐量反而低于 static_16,说明 batch size 过大有害

3. **实际 batch size 分布**
   
   **static_8 的分布: {2: 1, 6: 1, 8: 7}**
   - 大部分批次达到满 batch(8)
   - 少数批次因超时或清空而不足
   
   **static_16 的分布: {6: 1, 8: 1, 10: 5}**
   - 没有批次达到满 batch(16)
   - 平均实际 batch size 仅 9.14
   
   **static_32 的分布: {1: 1, 6: 1, 8: 1, 9: 1, 10: 4}**
   - 完全没有批次达到满 batch(32)
   - 平均实际 batch size 仅 8.00
   
   **关键洞察:**
   - 当 `arrival_rate = 10 req/s` 时,每 0.1s 到达一个请求
   - `flush_timeout = 0.5s` 意味着最多等待 5 个请求
   - 因此 static_16 和 static_32 永远无法凑满批次
   - 这解释了为什么 static_32 性能反而下降

4. **为什么连续批处理最优?**
   
   **静态批处理的瓶颈:**
   - 需要等待凑够 batch 或超时
   - 请求到达不均匀时,经常凑不满 batch
   - 完成的请求空占资源,直到整个批次完成
   
   **连续批处理的优势:**
   - 请求到达立即开始处理,无需等待
   - 完成的请求立即释放资源,新请求填补
   - 始终保持 GPU 满载运行
   
   **数学分析:**
   - 平均 batch size ≈ arrival_rate × avg_generation_time
   - avg_generation_time ≈ 1.0s (从结果推断)
   - 理论 batch size ≈ 10 × 1.0 = 10
   - 实际 static_16 的平均 batch size = 9.14,符合预期

5. **参数对结果的影响**
   
   - **arrival_rate = 10**: 决定了请求到达的密集程度
     - 如果 arrival_rate = 1,静态批处理会更差(更难凑满 batch)
     - 如果 arrival_rate = 100,静态批处理会接近最优
   
   - **flush_timeout = 0.5**: 静态批处理的等待时间
     - 增大 timeout 可以凑满更大的 batch,但增加延迟
     - 减小 timeout 可以降低延迟,但吞吐量下降
   
   - **max_tokens = 256**: 长序列放大了连续批处理的优势
     - 短序列时,请求完成快,连续批处理优势不明显

### 教学要点
- 连续批处理是 vLLM 的核心创新,解决了静态批处理的根本缺陷
- 静态批处理的性能高度依赖请求到达模式
- 连续批处理在各种到达模式下都能保持最优性能

---

## 技术实现细节

### 静态批处理的刷新机制
```python
# 三种触发刷新的条件
if len(pending) >= batch_size:
    should_flush = True  # 队列已满
elif pending and (now - pending[0].arrival_time) >= flush_timeout:
    should_flush = True  # 等待超时
elif i >= n and pending:
    should_flush = True  # 清空队列
```

### 连续批处理的核心循环
```python
while i < n or active or not llm.is_finished():
    # 注入已到达的请求
    while i < n and requests[i].arrival_time <= now:
        llm.add_request(req.prompt, sp)
    
    # 执行一步推理
    outputs, _ = llm.step()
    
    # 收集完成的请求
    for seq_id, token_ids in outputs:
        done.append(req)
```

### 异步处理机制
```python
# 使用 asyncio 实现时间控制
await asyncio.sleep(0.001)  # 短暂等待,避免忙等待
await asyncio.sleep(0)      # 让出控制权,允许其他协程运行
```

---

## 实验总结

### 静态批处理 vs 连续批处理

---

## 实验三: Chunked Prefill (`chunked_prefill.py`)

### 实验目的
验证长 prompt 与短 prompt 混合时，chunked prefill 是否能降低短请求延迟，缓解 decode 被长 prefill 长时间阻塞的问题。

### 核心思路
- 基线策略：完整 prompt 一次性 prefill
- Chunked Prefill：每次只处理 `prefill_chunk_size` 个 prompt token
- 当系统同时存在 `waiting` 与 `running` 请求时，调度器按 step 在 prefill chunk 与 decode 间交替

### 运行方式
```bash
python experiments/batching/chunked_prefill.py \
  --model /path/to/Qwen3-0.6B \
  --prefill-chunk-sizes 128 256 512 1024
```

默认会开启 `enforce_eager=True`，以便把实验聚焦在调度行为而不是 CUDA graph 形状限制上。如需恢复 CUDA graph，可追加 `--allow-cudagraph`。

脚本会固定先跑一组 `baseline`，再依次 sweep 你传入的 `chunk_size` 列表，方便直接比较 sweet spot。

### 默认工作负载
- 2 个超长 prompt 在 `t=0` 到达
- 12 个短 prompt 在后续 0.03s 间隔内陆续到达
- 长 prompt 默认约 2048 个词，短 prompt 默认约 32 个词
- 重点观察 `TTFT`，尤其是 `short P95 TTFT`

### 指标说明
- `Avg TTFT` / `P95 TTFT`: 所有请求首 token 时间
- `Short TTFT` / `Short P95`: 仅统计短请求的首 token 时间
- `Avg Lat` / `Short Lat`: 请求完整完成延迟

需要注意：`chunked prefill` 的核心目标是改善混合负载下的公平性和首 token 响应，因此最重要的指标不是 completion latency，而是 `TTFT`，尤其是短请求的 `P95 TTFT`。

### Sweep 实验结果

在 `Qwen3-0.6B`、默认工作负载、`enforce_eager=True` 下，运行：

```bash
python experiments/batching/chunked_prefill.py \
  --model /path/to/Qwen3-0.6B \
  --prefill-chunk-sizes 128 256 512 1024
```

得到结果：

```text
Strategy             Total(s)      Req/s   Avg TTFT   P95 TTFT  Short TTFT  Short P95    Avg Lat  Short Lat
baseline                1.154      12.13      0.107      0.235       0.085      0.191      0.813      0.757
chunk_128               2.757       5.08      0.366      1.919       0.105      0.115      1.643      1.458
chunk_256               1.816       7.71      0.210      0.972       0.083      0.110      1.179      1.073
chunk_512               1.277      10.96      0.132      0.423       0.081      0.090      0.868      0.801
chunk_1024              1.207      11.60      0.105      0.278       0.076      0.126      0.862      0.804
```

### 结果分析

**1. `chunk_size` 太小会让切换成本淹没收益**

- `chunk_128` 的 `Short P95` 从 baseline 的 `0.191s` 降到 `0.115s`
- 但同时吞吐从 `12.13 req/s` 掉到 `5.08 req/s`
- `Avg Lat` 也从 `0.813s` 上升到 `1.643s`

这说明过小的 chunk 虽然给了短请求更多插队机会，但会让 prefill/decode 交替过于频繁，固定调度和执行开销被显著放大。

**2. `chunk_256` 仍然偏碎，但已经开始体现收益**

- `Short P95` 进一步改善到 `0.110s`
- `Short TTFT` 也优于 baseline：`0.083s vs 0.085s`
- 但整体吞吐和 completion latency 仍然偏差

这说明 `256` 已经可以显著降低短请求尾延迟，但对整体效率的影响仍然较大。

**3. `chunk_512` 是这组 workload 下最均衡的点**

- `Short P95` 为 `0.090s`，相较 baseline 的 `0.191s` 改善最明显
- `Short TTFT` 也优于 baseline：`0.081s vs 0.085s`
- 同时吞吐仍保持在 `10.96 req/s`
- `Avg Lat` 仅从 `0.813s` 增加到 `0.868s`

这说明 `512` 既能有效给短请求让路，又没有把长请求切得过碎，是这一组实验中的 sweet spot。

**4. `chunk_1024` 更接近 baseline 行为**

- 吞吐和平均延迟都更接近 baseline
- `Short TTFT` 最好：`0.076s`
- 但 `Short P95` 回升到 `0.126s`，明显不如 `512`

这意味着 `1024` 虽然减少了切换开销，但短请求插入的频率也下降了，所以平均情况很好，尾部收益却不如更细粒度的 `512`。

### 结论

- 如果目标是改善短请求尾延迟，优先关注 `Short P95 TTFT`
- 在当前实现和这组 workload 下，`chunk_size=512` 是最推荐的默认值
- `chunk_size=1024` 适合更偏向吞吐、同时保留一定公平性收益的场景
- `chunk_size=128` 明显过小，不建议作为默认配置
- `chunk_size=256` 可作为保守配置，但综合表现不如 `512`

### 为什么 `512` 更合适？

- 当前实现是 step 级的 `prefill <-> decode` 交替，不是单个 CUDA batch 内真正混合，所以 chunk 越小，切换成本越大
- chunk 越大，短请求插队机会越少，尾延迟收益会下降
- `512` 正好落在“切换不过于频繁，同时仍能有效给短请求让路”的平衡点上

### 具体实现细节

这一版 `chunked prefill` 没有把 prefill 和 decode 混到同一个 CUDA batch 里，而是尽量在不大改现有执行栈的前提下，实现了 step 级交替调度。核心改动分布在 `Sequence`、`Scheduler`、`ModelRunner` 和 `BlockManager` 四个模块。

**1. `Sequence` 负责记录 prompt 的 prefill 进度**

- 新增 `prefill_chunk_size`，表示当前 step 准备处理多少个 prompt token
- 新增 `num_prompt_tokens_remaining` 和 `is_prefill_finished` 两个属性
- `num_cached_tokens` 继续作为“已经写入 KV cache 的 prompt token 数量”

这样调度器就不需要额外维护一套 prefill 进度表，序列对象本身就能表达“还剩多少 prompt 没算完”。

**2. `Scheduler` 在 step 级别交替 prefill / decode**

调度入口仍然是 `schedule()`，但在开启 `chunked prefill` 且系统中同时存在：

- `waiting` 请求
- `running` 请求

时，调度器会通过 `_schedule_prefill_next` 在两个 phase 间轮转：

```text
prefill(chunk) -> decode(step) -> prefill(chunk) -> decode(step)
```

具体来说：

- `_schedule_prefill()` 不再默认吞掉整条 prompt，而是只给当前序列分配一个 `prefill_chunk_size`
- 如果一个请求的 prompt 还没算完，它会被重新放回 `waiting` 队尾
- 如果该请求的 prompt 已经全部 prefill 完成，它才会进入 `running` 队列参与 decode

这里“放回队尾”是关键。如果未完成的长请求始终留在队首，短请求仍然会被挡住，虽然叫 chunked prefill，但实际上没有真正获得公平性收益。

**3. `ModelRunner.prepare_prefill()` 只准备当前 chunk**

传统 prefill 会直接把 `seq[num_cached_tokens:]` 全部送进模型；现在只准备：

```python
chunk_start = seq.num_cached_tokens
chunk_end = chunk_start + seq.prefill_chunk_size
```

对应的输入包括：

- `input_ids[chunk_start:chunk_end]`
- 对应的 `positions`
- 当前 chunk 每个 token 的 `slot_mapping`
- 以及需要时的 `block_tables`

这样模型前向只计算当前 chunk，而不是整条剩余 prompt。

**4. `postprocess()` 先更新 KV，再决定是否产出首 token**

在 prefill step 结束后：

- 先把 `seq.num_cached_tokens += seq.prefill_chunk_size`
- 如果 prompt 还没 prefill 完，就直接返回，不追加生成 token
- 只有最后一个 prefill chunk 完成后，才会真正 `append_token()` 生成首个输出 token

这与 decode 的行为不同。decode 每一步都会追加一个 token，而 prefill 的中间 chunk 只是推进 prompt 处理进度，不应该向用户暴露“半成品输出”。

**5. `BlockManager` 延迟注册 prefix cache**

这是 chunked prefill 下一个很容易忽略，但很重要的点。

在原始整段 prefill 语义里，`allocate()` 之后很快就会把整块 prompt 写入 KV cache；但在 chunked prefill 下，一个 block 可能只被处理了一部分。如果这时就把它注册进 prefix cache：

- 其他请求可能会错误复用一个“已经分配，但 KV 还没真正写完”的 block

所以现在的策略是：

- `allocate()` 时先分配 block
- 只有当某个完整 block 真的完成 prefill 后，才通过 `cache_full_blocks()` 写入哈希表

这样 prefix cache 的可见性和 KV 的真实完成状态是一致的。

**6. 为什么实验默认开启 `enforce_eager=True`**

当前 decode 的 CUDA graph 是按 capture 时的最大 `block_tables` 宽度预分配的，而 `chunked prefill` 实验里我们故意构造了更长上下文、更多动态变化的场景。为了把实验重点放在 scheduler 行为上，而不是被 graph shape 限制干扰，实验脚本默认禁用了 CUDA graph。

如果需要，也可以通过 `--allow-cudagraph` 恢复图执行，但更适合在固定上下文形状的场景下使用。

### 实现总结

这套实现的本质可以概括为：

- `Sequence` 追踪进度
- `Scheduler` 负责轮转
- `ModelRunner` 只算当前 chunk
- `BlockManager` 保证 prefix cache 正确性

它不是 vLLM 那种“单 batch 内 prefill + decode 真混合”的完整实现，但已经足以复现 chunked prefill 最关键的收益：用少量吞吐损失，换取混合负载下更好的短请求首 token 公平性。

### 依赖
- `nanovllm`: 核心 vLLM 实现
- `asyncio`: 异步IO框架
- `dataclasses`: 数据类支持
- Python >= 3.10
