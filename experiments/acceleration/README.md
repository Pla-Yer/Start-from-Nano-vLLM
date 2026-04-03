# Acceleration 实验

本目录包含针对推理加速技术的性能基准测试,通过对比 Eager 模式与 CUDA Graph 模式,深入理解 GPU 执行优化的原理。

## 实验目的
理解 CUDA Graph 优化技术如何通过减少 CPU 开销来提升推理性能。

## 核心概念

### Eager 模式 (enforce_eager=True)
PyTorch 的默认执行模式:
- 每次推理时,CPU 逐个发射 CUDA kernel 到 GPU
- CPU 需要等待每个 kernel 的调度完成
- 存在大量的 CPU-GPU 同步开销

### CUDA Graph 模式 (enforce_eager=False)
CUDA 的优化执行模式:
- 首次推理时捕获整个计算图(包括 kernel 序列和依赖关系)
- 后续推理直接重放捕获的计算图
- CPU 只需发射一次"重放"命令,大幅减少 CPU 开销

## 测试参数
```python
max_tokens = 64              # 每个请求生成64个token
num_repeats = 4              # 8个基础提示 × 4次重复 = 32个请求
temperature = 0.7            # 采样温度
warmup_prompts = 2           # 预热请求数
```

## 实验结果 (Qwen3-0.6B)
```
[eager / enforce_eager=True]
  Total time: 0.710s
  Throughput (tok/s): 2882.96
  Throughput (req/s): 45.05

[cudagraph / enforce_eager=False]
  Total time: 0.286s
  Throughput (tok/s): 7163.63
  Throughput (req/s): 111.93

[improvement: cudagraph vs eager]
  Time reduction: 59.8%
  Throughput increase: 148.5%
```

## 结果分析

### 关键发现: CUDA Graph 带来巨大性能提升

1. **性能提升显著**
   - 时间减少 59.8% (0.710s → 0.286s)
   - Token 吞吐量提升 148.5% (2883 → 7164 tok/s)
   - 请求吞吐量提升 2.5 倍 (45 → 112 req/s)

2. **为什么提升这么大?**
   
   **Eager 模式的开销来源:**
   - 每个 token 生成需要多次 kernel 发射
   - 对于 64 token 生成,可能需要数百次 kernel 发射
   - 每次 kernel 发射都有 CPU-GPU 通信开销
   
   **CUDA Graph 的优化:**
   - 首次推理捕获计算图(约 1-2 秒)
   - 后续推理只需一次"重放"命令
   - CPU 开销从 O(num_kernels) 降到 O(1)
   
   **加速比分析:**
   - 理论上,CUDA Graph 可以消除所有 CPU 调度开销
   - 实际加速比 = 7163.63 / 2882.96 = 2.48 倍
   - 说明 CPU 调度开销在 Eager 模式中占比约 60%

3. **参数对结果的影响**
   
   - **max_tokens = 64**: 序列越长,CUDA Graph 收益越大
     - 如果 max_tokens = 16,加速比会减小(因为 kernel 发射次数少)
     - 如果 max_tokens = 256,加速比可能更大
   
   - **批处理大小**: 批处理越大,每个 batch 的 kernel 发射次数越多
     - CUDA Graph 在大批处理场景下收益更明显
   
   - **模型复杂度**: 模型越复杂,每步推理的 kernel 数量越多
     - 大模型从 CUDA Graph 中获益更多

4. **CUDA Graph 的代价**
   
   - **初始化开销**: 首次推理需要捕获计算图(额外 1-2 秒)
   - **内存开销**: 需要存储捕获的计算图
   - **灵活性限制**: 不支持动态控制流(if/while 等动态操作)

## 技术实现细节

### 预热机制
```python
# 正式测试前运行2个请求预热
warmup_batch = prompts[:2]
_ = llm.generate(warmup_batch, sp, use_tqdm=False)

# CUDA Graph 模式额外等待1秒,确保图捕获稳定
if not enforce_eager:
    time.sleep(1)
```

**原因:**
- 消除模型加载、CUDA 初始化等首次开销
- CUDA Graph 捕获需要稳定的 GPU 状态

### 资源清理机制
```python
def cleanup():
    # 销毁分布式进程组
    if dist.is_initialized():
        dist.destroy_process_group()
    
    # 同步 CUDA 设备
    torch.cuda.synchronize()
    
    # 清理垃圾回收和 CUDA 缓存
    gc.collect()
    torch.cuda.empty_cache()
```

**原因:**
- 确保每个测试用例的公平性
- 避免资源泄漏影响后续测试

### 子进程隔离
```python
# 每个测试模式在独立子进程中运行
subprocess.run([sys.executable, script_path, 
                "--model", model_path, 
                "--mode", mode])
```

**原因:**
- CUDA Graph 捕获后难以释放
- 子进程隔离确保测试独立性

## 实际应用场景

### 适合 CUDA Graph 的场景
1. **在线推理服务**: 请求量大,CPU 开销占比高
2. **长序列生成**: token 生成次数多,kernel 发射频繁
3. **批处理推理**: 大批处理增加 kernel 发射次数

### 不适合 CUDA Graph 的场景
1. **动态控制流**: 需要根据输入动态改变计算图
2. **单次推理**: 初始化开销无法摊销
3. **内存受限**: 无法承担额外的内存开销

## 教学要点

1. **CPU 开销不可忽视**
   - 在 Eager 模式下,CPU 调度开销可达总时间的 60%
   - GPU 利用率受限于 CPU 的 kernel 发射速度

2. **CUDA Graph 的本质**
   - 将"解释执行"变为"编译执行"
   - 类似于 JIT 编译的思想

3. **性能优化的权衡**
   - CUDA Graph 用初始化时间和内存换取运行时性能
   - 需要根据实际场景评估是否值得

## 参数调优建议

1. **默认启用 CUDA Graph**: 对于大多数推理场景,收益远大于代价
2. **注意初始化时间**: 首次推理会慢,需要预热
3. **监控内存使用**: 确保有足够内存存储计算图

## 依赖
- `nanovllm`: 核心 vLLM 实现
- `torch`: PyTorch 框架
- `torch.distributed`: 分布式训练模块
- Python >= 3.10
