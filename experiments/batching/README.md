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

| 维度 | 静态批处理 | 连续批处理 |
|------|-----------|-----------|
| 吞吐量 | 依赖 batch size 和到达模式 | 始终最优 |
| 延迟 | 受排队时间影响大 | 始终最低 |
| P95 延迟 | 波动大 | 稳定 |
| 资源利用率 | 有空闲期 | 始终满载 |
| 实现复杂度 | 简单 | 复杂 |

### 参数调优建议

1. **静态批处理**
   - 需要根据请求到达模式调整 batch size
   - `flush_timeout` 需要在吞吐量和延迟间权衡
   - 适合离线批处理场景

2. **连续批处理**
   - 无需调参,自适应各种场景
   - 适合在线服务场景
   - 是 vLLM 的默认和推荐策略

### 依赖
- `nanovllm`: 核心 vLLM 实现
- `asyncio`: 异步IO框架
- `dataclasses`: 数据类支持
- Python >= 3.10
