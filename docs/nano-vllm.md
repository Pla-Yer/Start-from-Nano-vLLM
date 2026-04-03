# nano-vllm

## LLMEngine：

### 初始化：创建张量并行模型进程

多进程：torch.multiprocessing as mp

```python
# 使用spawn进程创建
ctx = mp.get_context("spawn")
...

for(i in range(config.tensor_parallel_size))
    #创建通信事件
    event = ctx.Event()
    #创建进程
    process = Process（target=ModelRunner,args=(config,i,event)）
...
#初始化ModeRunner，modelrunner主要
self.model_runner = ModelRunner(config, 0, self.events)
#初始化schedule
self.scheduler = Scheduler(config)
```

#### modelrunner：

初始化模型并加载：

```
      # 初始化模型
        self.model = Qwen3ForCausalLM(hf_config)
        # 加载预训练权重
        load_model(self.model, config.model)
```

初始化采样器：

```
# 初始化采样器
        self.sampler = Sampler()
```

预热模型：将除内存分配外的模型流程走一遍检测是否正确初始化

```
        # 预热模型
        self.warmup_model()
```

分配kv缓存：通过计算将剩余空间全部划分为block并分配到每个隐藏层中

```
# 分配KV缓存
        self.allocate_kv_cache()
...
#核心操作：
# 创建KV缓存张量，这里占用了分配的所有可用显存，
# 此时通过nvidia-smi可以发现显存此时才被大规模暂用
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        # 将KV缓存分配给模型的每一层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):  # 如果模块有K和V缓存
                module.k_cache = self.kv_cache[0, layer_id]  # K缓存
                module.v_cache = self.kv_cache[1, layer_id]  # V缓存
                layer_id += 1
```

捕获graph 图

```
        # 如果不禁用CUDA图优化，则捕获CUDA图
        if not self.enforce_eager:
            self.capture_cudagraph()
```

创建共享内存（如果使用张量并行）

```python
# 如果使用张量并行
        if self.world_size > 1:
            if rank == 0:  # 主进程
                # 创建共享内存用于进程间通信
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # 同步所有进程
            else:  # 从进程
                dist.barrier()  # 同步所有进程
                # 连接到共享内存
                self.shm = SharedMemory(name="nanovllm")
                # 从进程进入循环等待主进程的命令
                self.loop()
```

#### scheduler：初始化调度器

```python
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
```

核心在于初始化了blockmanager和两个队列waiting和running，waiting表示该队列中的seq需要进行prefill，running表示该队列中的seq需要进行decoder。

##### BlockManager：初始化块调度器

```python
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

核心在于初始化了队列freeblock和集合usedblocks已经BLock类的list，每个block类里面存储了：

```python
        self.block_id = block_id#id
        self.ref_count = 0#引用次数（用于清理）
        self.hash = -1#哈希值，用于查找
        self.token_ids = []#块内的tokens的id（用于判断缓存命中）
```

### generate：调用入口

#### add_request：将prompt转化为Sequence对象并加入到外套队列中

```python
# 1. 添加请求
def add_request(self, prompt, sampling_params):
    # 将字符串转为 token IDs
    prompt = self.tokenizer.encode(prompt)
    # 创建 Sequence 对象
    seq = Sequence(prompt, sampling_params)
    # 加入 waiting 队列
    self.scheduler.add(seq)
```

Sequence：为seq添加对象信息与方法，为后面的处理做准备，包括：

```
        - seq_id: 序列的唯一标识符
        - status: 序列当前状态（WAITING/RUNNING/FINISHED）
        - token_ids: 包含prompt和已生成token的完整token ID列表
        - last_token: 最新生成的token ID
        - num_tokens: 序列中token的总数（prompt + 已生成的token）
        - num_prompt_tokens: 原始prompt的token数量
        - num_cached_tokens: 已经被缓存到KV缓存中的token数量
        - block_table: 块表，记录序列使用的物理块索引
        - temperature: 采样温度，控制生成随机性
        - max_tokens: 最大生成token数（包括prompt）
        - ignore_eos: 是否忽略EOS token，继续生成
        - num_tokens：序列的总长度
        - is_finished：序列是否处理完成
        - num_completion_tokens：生成的token数
        - prompt_token_ids：prompt的token序列
        - completion_token_ids：生成内容的token序列
        - num_cached_blocks：已经缓存了的块数
        - num_blocks：序列需要的块数
        - last_block_num_tokens：最后一个块的token数
```

#### step：调度与执行

```python
  def step(self):
          #通过schedule，判断当前需要处理的序列与对应的任务（prefill/decoder）
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens
```

### schedule：

将工作分为填充（prefill）与解码（decoder），优先做prefill。每次运行，只会做prefill，decoder两者中的一个工作。

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # ===== 第一阶段：Prefill =====
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    # 优先处理prefill任务
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

对于填充过程。包括以下步骤：

1. 资源判断：两个约束：批次最大token、可分配block数。

批次最大token主要作用于高请求时进行有效限制，可分配block数用于实际系统可使用内存。这样设计是因为，prefill过程是高负载场景，通过限制单批次同时输入，可以更有效地利用计算能力与显存，所以这个约束不会在decoder使用。

   2. 分配资源：self.block_manager.allocate(seq)

```python
    def allocate(self, seq: Sequence):
        """
        为序列分配KV缓存块
        此方法将为序列的每个块分配物理存储空间，并尝试利用已存在的块（前缀缓存）

        参数:
        - seq: 需要分配块的序列对象
        """
        assert not seq.block_table  # 确保序列还没有分配块表
        h = -1  # 哈希值，用于前缀缓存匹配
        cache_miss = False  # 标记是否发生缓存未命中

        # 遍历序列需要的所有块
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # 获取第i个块的token IDs
            # 计算哈希值，只有当块是满的（等于block_size）时才计算，否则设为-1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            # 尝试从哈希表中查找相同内容的块
            block_id = self.hash_to_block_id.get(h, -1)

            # 如果找不到对应的块或内容不匹配，则发生缓存未命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            # 处理缓存未命中情况
            if cache_miss:
                # 从空闲块列表中获取一个块ID
                block_id = self.free_block_ids[0]
                # 分配该块并更新状态
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：增加已缓存的token数量
                seq.num_cached_tokens += self.block_size
                # 检查块是否已在使用中
                if block_id in self.used_block_ids:
                    # 如果块已在使用中，增加引用计数而不重新分配
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 如果块不在使用中，分配该块
                    block = self._allocate_block(block_id)

            # 如果哈希有效，更新块内容和哈希表
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            # 将块ID添加到序列的块表中
            seq.block_table.append(block_id)
```

根据seq的信息，为seq的每个block分配blockid，如果对于某个block已经被分配了那就不重复添加，没有的就分配空闲的block。这里主要是逻辑上的分配，将seq的tokens分块，每块对应一个block对象，在blocks的list中通过blockid将seq与block对象建立映射（seq.block_table->blockid->blocks->block）。

这里还有一个查找缓存命中的功能，当且仅当一个block满了的时候，才会被查找，如果新的req的token和block中的token对应上，那么就会复用它（ref_count+1），这个功能主要是用于系统提示词（system_prompt）的cache复用。

3. 状态更新：

```python
            seq.status = SequenceStatus.RUNNING#seq状态变为decoder
            self.waiting.popleft()#将该seq从waiting队列中移除
            self.running.append(seq)#加入running队列
            scheduled_seqs.append(seq)#加入scheduled_seqs list，用于发给计算模块进行计算
```

对于解码过程，包括以下操作：

```python
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
```

1. 是否进行decoder判断：有无需要decoder的seq，当前seq数要小于设定的上限

2. 当前seq出队

3. 资源判断

```python
def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

这个判断是基于block进行的判断，因为cache是安装block进行分配的，如果seq的大小刚好为块大小*n+1意味着此时需要增加一个块，然后再判断空闲块是否大于1（是否有空闲块），便可以判断，下一token是否可以进行decoder。

4. 是否抢占

如果没有出现需要新的block但没有空闲block的情况，那么自然进行，将seq装入scheduled_seqs送给计算模块。但是如果出现资源满载的情况，那么就会进行抢占

```python
def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    def deallocate(self, seq: Sequence):
        """
        释放序列占用的KV缓存块
        此方法会减少每个块的引用计数，当引用计数为0时，将块标记为空闲

        参数:
        - seq: 需要释放块的序列对象
        """
        # 遍历序列的块表，逆序处理
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]  # 获取块对象
            block.ref_count -= 1  # 减少引用计数
            # 如果引用计数为0，说明没有其他序列在使用此块，可以释放
            if block.ref_count == 0:
                self._deallocate_block(block_id)  # 释放该块

        # 重置序列的缓存统计信息
        seq.num_cached_tokens = 0  # 清零已缓存的token数量
        seq.block_table.clear()  # 清空块表
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
```

首先看是否有正在运行中的seq，如果有，那么就把它状态改为waiting，再将它的每个blockid对应的block对象的引用计数减一，引用为0则将其block对象进行清除，将它的block退出used_blocks,进入free_blocks。

如果没有，那么说明该seq过长，当前模型不能处理，这里将该seq仍然放回waiting队列是一个不明智的做法。

### run：

```python
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        执行模型推理

        参数:
        - seqs: 序列列表
        - is_prefill: 是否为prefill阶段

        返回:
        - 生成的token ID列表
        """
        # 根据阶段准备输入数据
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 准备采样参数
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 运行模型获得logits
        logits = self.run_model(input_ids, positions, is_prefill)
        # 采样生成token
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()  # 重置上下文
        return token_ids
```

通过简明的接口，指明数据流向：prepare->run_model(tansformer)->sampler（softmax）。

#### prepare：准备context

对于模型来说，计算过程是不变的，但是对于prefill与decoder两种任务来说，怎么样计算却是不同的，通过prepare过程，让run_model过程可以适应不同的任务。

为了实现上面的的要求，我们通过context的全局变量实现接口同一，但是内容不同，即通过修改_CONTEXT类的数据，不显式传输非数据内容。

```python
class Context:
    is_prefill: bool = False#是否是prefill任务
#当前batch的查询序列长度累积值，用于引导扁平化后的新 token（Q）如何按序列切分
    cu_seqlens_q: torch.Tensor | None = None
#当前batch中键序列长度累积值，用于每个序列 attention 时实际可见的 K/V 长度边界
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0#当前batch中的最大查询序列（q）长度，用于将每个seq的q对齐（补全到max）
    max_seqlen_k: int = 0#最大键序列长度（k），同上
#槽位映射列表，将每个seq的新生成的kv的存储位置映射，list[list]
    slot_mapping: torch.Tensor | None = None
#下文长度列表，在decoder阶段不需要cu_seqlens_k与cu_seqlens_q，只需要传递总长度即可
    context_lens: torch.Tensor | None = None
#当存在之前的block时，需要告诉attention那些block中的kv会被用到
    block_tables: torch.Tensor | None = None
```

##### prepare_prefill

```python
    def prepare_prefill(self, seqs: list[Sequence]):
        """
        为prefill阶段准备输入数据

        参数:
        - seqs: 序列列表

        返回:
        - input_ids: 输入token ID张量
        - positions: 位置编码张量
        """
        input_ids = []  # 输入token ID列表
        positions = []  # 位置编码列表
        cu_seqlens_q = [0]  # 查询序列长度累积值
        cu_seqlens_k = [0]  # 键序列长度累积值
        max_seqlen_q = 0  # 最大查询序列长度
        max_seqlen_k = 0  # 最大键序列长度
        slot_mapping = []  # 槽位映射列表
        block_tables = None  # 块表

        for seq in seqs:
            seqlen = len(seq)  # 序列长度
            # 添加未缓存的token ID
            input_ids.extend(seq[seq.num_cached_tokens:])
            # 添加对应的位置编码
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens  # 查询序列长度
            seqlen_k = seqlen  # 键序列长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:    # 预热阶段，跳过
                continue

            # 为每个块中的槽位建立映射，将seq需要被缓存的位置映射到对应的物理块位置
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:  # 不是最后一个块
                    end = start + self.block_size
                else:  # 最后一个块
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 说明存在之前的缓存token，需要使用前缀缓存
            block_tables = self.prepare_block_tables(seqs)

        # 创建CUDA张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 设置上下文
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions
```

输出input_ids = [seq1_new_tokens | seq2_new_tokens | seq3_new_tokens ...]是需要计算kv的新的token的，但是多seq拼起来的，所以也需要position来明确这些token在原seq中的位置。input_ids.length=position.length。

显然，也需要cu_seqlens_q来明确哪些tokens是哪个seq的，而在decoder时需要seq的全部来推理下一个，所以需要`cu_seqlens_k`来明确每个seq的总长度，需要注意的是，这两个都是通过累加来传递的，cu_seqlens_q=[0 5 12]表示第一个seq有5个token来计算kv，第二个seq有12-5=7个token来生成新kv。当cu_seqlens_q=cu_seqlens_k，说明当前batch中，所有seq都是新的，没有复用任何已有kv cache。**显而易见，这两者都是prefill独有的，因为decoder只需要生成下一个即可**

##### prepare_decoder:

```python
def prepare_decode(self, seqs: list[Sequence]):
        """
        为decode阶段准备输入数据

        参数:
        - seqs: 序列列表

        返回:
        - input_ids: 输入token ID张量
        - positions: 位置编码张量
        """
        input_ids = []  # 输入token ID列表
        positions = []  # 位置编码列表
        slot_mapping = []  # 槽位映射列表
        context_lens = []  # 上下文长度列表

        for seq in seqs:
            input_ids.append(seq.last_token)  # 添加最新token
            positions.append(len(seq) - 1)  # 添加位置编码
            context_lens.append(len(seq))  # 添加上下文长度
            # 计算槽位映射
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)

        # 创建CUDA张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 准备块表
        block_tables = self.prepare_block_tables(seqs)

        # 设置上下文，这里将上面的计算结果写为全局变量，便于底层算子快速访问，同时模型输入与元数据分开，接口更加清晰
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions
```

为decoder准备上下文CONTEXT与input_ids与对应的position，更简单，因为确定了每次生成一个，只需要每次更新当前seq的长度即可，而且在schedule阶段，已经处理了当前token需要新block的分配任务，这里只需要在seq的block_table的最后一个block里面添加上即可。

而且因为decoder阶段一定有之前的kv cache，所以直接分配就行，不需要判断。

#### run_model: 模型推理

```python
@torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        运行模型推理

        参数:
        - input_ids: 输入token ID张量
        - positions: 位置编码张量
        - is_prefill: 是否为prefill阶段

        返回:
        - 模型输出logits
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 如果是prefill阶段、强制使用eager模式或批大小过大，则直接运行模型
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 使用CUDA图优化
            bs = input_ids.size(0)  # 批大小
            context = get_context()  # 获取上下文
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]  # 获取合适的CUDA图
            graph_vars = self.graph_vars  # 获取图变量

            # 更新图变量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # 重放CUDA图
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])
```

分为两种执行方式：eager，cuda graph

eager就是标准的模型推理模式，由cpu全程控制参数的流动；cuda graph模式则是将eager模式中重复标准的部分记录下来，全程由GPU执行，无需CPU参与调度。

##### 标准推理：

```python
class Qwen3ForCausalLM(nn.Module):
    """
    用于因果语言建模的Qwen3模型
    包含完整的模型结构和语言模型头
    """

    # 定义打包模块的映射关系，用于处理模型权重的加载
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),      # 查询投影对应到QKV投影中的Q部分
        "k_proj": ("qkv_proj", "k"),      # 键投影对应到QKV投影中的K部分
        "v_proj": ("qkv_proj", "v"),      # 值投影对应到QKV投影中的V部分
        "gate_proj": ("gate_up_proj", 0), # 门控投影对应到合并列投影中的第0部分
        "up_proj": ("gate_up_proj", 1),   # 上投影对应到合并列投影中的第1部分
    }

    def __init__(
        self,
        config: Qwen3Config  # 模型配置
    ) -> None:
        super().__init__()
        # 主要的Qwen3模型
        self.model = Qwen3Model(config)
        # 并行的语言模型头，用于将隐藏状态转换为词汇表概率
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # 如果配置中设置了词汇嵌入和LM头权重共享，则绑定它们
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,   # 输入token ID
        positions: torch.Tensor,   # 位置索引
    ) -> torch.Tensor:
        """
        前向传播，返回模型的隐藏状态
        :param input_ids: 输入token ID
        :param positions: 位置索引
        :return: 模型隐藏状态
        """
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
    ) -> torch.Tensor:
        """
        计算对数几率(logits)，即将隐藏状态转换为词汇表上的概率分布
        :param hidden_states: 模型的隐藏状态
        :return: logits张量
        """
        return self.lm_head(hidden_states)
```

将模型分为两个部分：model+head，详情参考 [qwen3](/home/ttt/PY080313/AI/nano-vllm/大语言模型结构-qwen3.md)

#### sampler：采样

```python
class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
```

先把 logits 按 temperature 调整，再变成概率分布，然后不用显式调用 `multinomial`，而是用一种等价的随机扰动方法完成采样。

##### 1. temperature

计算logits / temperature

当 T<1：分母更小，logits 差异被放大，分布更尖锐。

当 T>1：logits 差异被压缩，分布更平缓。

当 T→0：会越来越接近贪心选最大值。

通过 temperature 控制随机性。

##### 2. softmax

通过softmax函数将logits变为概率分布。

##### 3. 采样

```python
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

**对每个概率 pi​，除以一个独立的指数分布随机变量 Ei​，然后取最大值的索引。**  
这样得到的索引分布，等价于按 pi​ 进行一次 categorical sampling。

如果 Ei​∼Exponential(1) 独立同分布：

$$
E_i \sim \text{Exponential}(1),\quad \text{i.i.d.}
$$

那么：

$$
i^* = \argmax_i \frac{p_i}{E_i}
\quad\Rightarrow\quad
\mathbb{P}(i^*=i) = p_i
$$

也就是说，它和直接从 categorical distribution p 中采样是等价的。

对比torch.multinomial

![](/home/ttt/.config/marktext/images/2026-03-30-17-14-08-image.png)

**1. 全并行，消灭串行瓶颈**

`torch.multinomial` 内部需要先做 `cumsum`（前缀和），这是一个有数据依赖的串行操作——每个元素必须等前一个算完。而 Gumbel-max 全程都是逐元素操作（`exponential_` 采样 + `div_`）加上一个 `argmax` 归约，这些在 GPU 上都有高度优化的 parallel reduction 实现，没有串行依赖。

**2. 数值稳定性更好**

`multinomial` 依赖累加浮点数，词表很大时（如 128K tokens）float32 的精度误差会积累在 CDF 尾部，导致低概率 token 的采样分布偏差。Gumbel-max 每个 token 独立操作，不存在累加误差传递。

**3. 原地操作，减少内存分配**

代码中的 `probs.div_()` 和 `exponential_()` 都是 in-place 操作（下划线后缀），避免了中间 tensor 的分配与回收，显存压力更小，对 decode 阶段的每步延迟很敏感。

**4. `clamp_min_(1e-10)` 防止除零**

Exponential(1) 理论上可以产生极小值（趋近 0），除以极小数会产生 `inf`，影响 `argmax` 的正确性。这个 clamp 是数值保险，几乎不改变分布（因为对应的 pi​/ϵ→∞ 的概率本身极小）。

**5. 在 speculative decoding / batched sampling 场景中尤为突出**

当需要对一整个 batch 的 token 同时采样时，`argmax(dim=-1)` 天然支持 batch 维度，而 `multinomial` 在 batch 上的并行支持历史上有各种实现限制。

### schedule.postprocess

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

对seq序列进行后处理，包括两步：

1. 对每个seq，加上最后生成的token

2. 判断seq是否结束，条件为：最后一个token为EOS且设定为不忽略EOS，或到达序列最大生成长度。如果结束，更新seq的状态；将seq占用的block还回free；从running队列中移除。



## 完成后处理：

```python
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens
```

当seq完成后（seq.is_finished），将该seq放入outputs里面，num_tokens用于记录处理量来进行评估。

```python
def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
```

当一个seq完成时，output有内容，就直接放入outputs中，用seq_id来和输入prompt的顺序对应上。

当所有内容完成时，按seq_id排序，然后decoder为文本并整理为

- `"text"`：解码后的字符串
- `"token_ids"`：原始 token 序列

关闭进度条并返回。

最后读取output：

```python

```
