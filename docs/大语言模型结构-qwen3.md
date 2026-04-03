# 大语言模型结构-qwen3

## 图

![https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/img/encoder.png](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/img/encoder.png)

## Qwen3ForCausalLM

顶层：模型主干（Qwen3Model，stage1-4）+ 输出头（ParallelLMHead，stage5（linear））

```
# 主要的Qwen3模型
self.model = Qwen3Model(config)
 # 并行的语言模型头，用于将隐藏状态转换为词汇表概率
self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
```

### Qwen3Model（stage1-4）

调用`Qwen3DecoderLayer`，构成Transformer decoder（embedding（stage1）+decoderlayers（stage 3）+normal（stage 4））

```
        # 词汇表并行嵌入层，将token ID转换为向量
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 创建多个解码器层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终的归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

#### VocabParallelEmbedding(stage1)

并行运行embed，因为embed本质是一个“查找表”，输入的字符被tokenizer变为token id，通过embedding将token id映射到高维向量空间（类比cv中的特征提取，但是文字是确定的所以更贴切与查找表）

```python
class VocabParallelEmbedding(nn.Module):
    """
    词汇表并行嵌入层
    将词汇表分割到多个GPU上，每个GPU只存储部分词汇表的嵌入向量
    """

    def __init__(
        self,
        num_embeddings: int,  # 词汇表总大小
        embedding_dim: int,   # 嵌入维度
    ):
        super().__init__()
        # 获取分布式训练的rank和world size
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # 确保词汇表大小可以被并行度整除
        assert num_embeddings % self.tp_size == 0

        # 计算每个分片的词汇表大小
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size

        # 计算当前rank负责的词汇表范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 初始化当前rank的嵌入权重
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器，用于从完整权重中提取当前rank对应的分片
        """
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        x: 输入的token ids
        """
        if self.tp_size > 1:
            # 对输入的token id进行掩码，确定哪些属于当前rank的词汇表范围
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将token id转换为当前分片内的索引
            x = mask * (x - self.vocab_start_idx)
        # 使用F.embedding进行嵌入查找
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 对结果应用掩码，不属于当前分片的部分设为0
            y = mask.unsqueeze(1) * y
            # 通过all_reduce聚合所有rank的结果
            dist.all_reduce(y)
        return y
```

可以看见这里主要在做并行任务（将嵌入权重分配到不同gpu上，然后再汇总），核心任务只有一个：

```python
y = F.embedding(x, self.weight)
```

将输入的x变为高维（512）的向量表示。

#### Qwen3DecoderLayers（stage3,4）

在Transformer中decoder是很多重复的层，他们的区别只有参数，其他的完全一样，所以在model中，直接循环初始化：

```python
self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
```

单层DecoderLayer

```python
class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3解码器层
    包含自注意力机制和前馈网络，以及相应的层归一化
    """

    def __init__(
        self,
        config: Qwen3Config,  # 模型配置
    ) -> None:
        super().__init__()
        # 自注意力层
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # MLP前馈网络
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 输入层归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 注意力后的层归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,       # 位置索引张量
        hidden_states: torch.Tensor,   # 输入隐藏状态
        residual: torch.Tensor | None, # 残差连接张量，如果为None则表示是第一层
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        :param positions: 位置索引
        :param hidden_states: 输入隐藏状态
        :param residual: 残差连接张量
        :return: (新的隐藏状态, 新的残差张量)
        """
        # 如果没有残差连接，则创建一个新的残差(即输入本身)
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 否则，在归一化的同时保留残差连接
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # 应用自注意力机制
        hidden_states = self.self_attn(positions, hidden_states)

        # 在注意力后进行层归一化，并保持残差连接
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # 通过MLP前馈网络
        hidden_states = self.mlp(hidden_states)

        # 返回新的隐藏状态和残差张量
        return hidden_states, residual
```

这里是标准的Transformer架构，归一化->自注意力->归一化->MLP。

这里需要注意的是，显式维护了残差，而不是

```python
x = x + Attention(Norm(x))#这里的x就是残差，它等于上一阶段的输入+Layer的输出
x = x + MLP(Norm(x))
```

直接维护一个同一的x，因为传统的实现方式在底层意味着：中间需要add->写回，而这种双流接口写法，通常是为了支持更高效的融合实现，把：residual add与norm尽量合成更紧凑的计算流程，实现

- 减少中间张量
- 融合 residual add + norm
- 降低显存访问
- 适合被cuda graph捕获

###### RMSNorm

```python
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
```

$$
\text{RMSNorm}(x) = \text{weight} \odot \frac{x}{\sqrt{\text{mean}(x^2, \text{dim=-1}) + \epsilon}} 
$$

RMS归一化，相较于传统归一化，它没有减均值，只缩放了幅度。

在RMSNorm中，有下面的推理优化点：

1. 不仅计算了归一化，还计算了当前层的残差，这样做也正是为了更好的编译优化，接口紧凑；

2. 使用@torch.compile，在编译时会将函数内操作进行融合；

3. 还使用了mul_与add_，这表示会在原来的内存上进行乘加，减少内存占用；

4. 在进行Norm前，将精度转化为float32，完成后转回原精度，确保归一化的稳定性

##### attention

```python
class Qwen3Attention(nn.Module):
    """
    Qwen3模型的注意力层实现
    包含查询、键、值的投影以及注意力计算
    """

    def __init__(
        self,
        hidden_size: int,           # 隐藏层维度
        num_heads: int,             # 查询头的数量
        num_kv_heads: int,          # 键值对头的数量
        max_position: int = 4096 * 32,  # 最大位置编码
        head_dim: int | None = None,    # 头部维度，如果为None则根据hidden_size和num_heads计算
        rms_norm_eps: float = 1e-06,    # RMS Norm的epsilon值
        qkv_bias: bool = False,         # 是否使用QKV偏置
        rope_theta: float = 10000,      # RoPE的角度基值
        rope_scaling: tuple | None = None,  # RoPE缩放参数
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()  # 获取分布式训练的进程数量
        self.total_num_heads = num_heads  # 总查询头数
        assert self.total_num_heads % tp_size == 0  # 确保头数能被进程数整除
        self.num_heads = self.total_num_heads // tp_size  # 当前进程负责的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        assert self.total_num_kv_heads % tp_size == 0  # 确保KV头数能被进程数整除
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 当前进程负责的KV头数
        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 计算头部维度
        self.q_size = self.num_heads * self.head_dim  # 查询张量大小
        self.kv_size = self.num_kv_heads * self.head_dim  # 键值张量大小
        self.scaling = self.head_dim ** -0.5  # 注意力分数缩放因子
        self.qkv_bias = qkv_bias  # 是否使用QKV偏置

        # QKV投影层，将隐藏状态映射到查询、键、值空间
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影层，将多头注意力输出映射回隐藏状态维度
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 旋转位置编码(RoPE)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # 注意力计算模块
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # 如果不使用QKV偏置，则应用RMS归一化到查询和键
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,      # 位置索引张量
        hidden_states: torch.Tensor,  # 输入的隐藏状态
    ) -> torch.Tensor:
        """
        前向传播
        :param positions: 位置索引
        :param hidden_states: 输入隐藏状态
        :return: 注意力输出
        """
        # 对输入应用QKV投影
        qkv = self.qkv_proj(hidden_states)
        # 将QKV分割成查询、键、值三个部分
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 重塑形状为(batch_size, num_heads, head_dim)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # 如果不使用QKV偏置，对查询和键应用RMS归一化
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 计算注意力输出
        o = self.attn(q, k, v)
        # 应用输出投影并返回结果
        output = self.o_proj(o.flatten(1, -1))
        return output
```

对hidden_state做atten计算，hidden_states -> qkv_proj -> q,k,v -> rope -> attn -> o_proj

一次GEMM线性映射到qkv空间，优势：

- 接口紧凑

- 减少kernal

- 减少hidden_state的访问次数

- 便于融合与抓取cuda graph

使用GQA，一个kv，多个q，减少kv cache。用不同的 q head 数和 kv head 数把 GQA 真正落到张量形状上

这里负责准备atten计算前的工作，将atten的计算下沉到atten模块中

###### 1. qkv投影

```python
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

这里的核心点在于，将qkv的参数矩阵拼在一起，形成一个大矩阵，这样就可以实现一次计算便得到hidden_state的q,k,v投影向量，

```python
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
```

在ColumnParallelLinear中配置了tensor并行模式：将head放在不同的GPU上并行计算，这是因为：

- **每个 head 的 Q/K/V 投影天然对应输出通道的一个子块**
- **每个 head 的 attention 在输出投影前互不依赖**
- **这些子块可以直接映射到矩阵的不同输出分片**
- **矩阵乘法本来就适合按输出分片做并行**

$$
\begin{aligned}
Q &= H W_q^\top + b_q \\
K &= H W_k^\top + b_k \\
V &= H W_v^\top + b_v
\end{aligned}

$$

$$
W_{\text{qkv}} \in \mathbb{R}^{(d_q+d_k+d_v) \times d_{\text{model}}}
$$

$$
W_{\text{qkv}} = \begin{bmatrix} W_q \\ W_k \\ W_v \end{bmatrix}
$$

$$
Y = H W_{\text{qkv}}^\top + b_{\text{qkv}}
$$

**把三次独立的 GEMM 合并成一次大 GEMM，算完再切开，就是 QKV 融合的数学本质**。

###### 2.rotary_emb 旋转位置编码

```python
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
```

意思是说，先计算一个[n,1,head_dim]的查找表，n表示模型最长生成长度，然后通过position返回token的位置从查找表找到位置对应的向量（长度为head_dim），然后将该token的所有head向量分为前半与后半（x1,x2）与查找表的向量两半（cos,sin），用    y1 = x1 * cos - x2 * sin；y2 = x2 * cos + x1 * sin计算新的前半和后半，然后组合回去。

优化点：

- 查找表设计，减少重复计算，同时设计为[max_position, 1, head_dim]便于广播

###### 3. attention

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
```

这里是将准备好的qkv高效接上缓存并使用flashattention进行计算。包括以下步骤：

- **把新产生的 k/v 写入 KV Cache**
- **根据当前是 prefill 还是 decode，选择不同的 attention 内核，对于prefill，会将没有kv cache缓存的内容全部计算，而Decoder只会计算下一个**
- **把 context 里的调度信息传给 FlashAttention / KV cache 内核**

核心优化：

- ## KV Cache 独立写入
  
  `store_kvcache` 用 Triton 自己写，说明作者不想把“写 cache”这件事交给通用 PyTorch 操作，而是明确优化成：
  
  - 每 token 一次写入
  - 按 slot_mapping 定位
  - 直接写到目标物理位置
  
  这比 Python 循环或者笨重 indexing 更贴近高性能推理需求。

- ## prefill / decode 用不同内核
  
  这是推理系统最重要的优化之一。
  
  ### prefill
  
  - query 多
  - key/value 多
  - 更像大矩阵 attention
  - 用 `flash_attn_varlen_func`
  
  ### decode
  
  - query 很短，通常 1
  - key/value 很长，来自历史 cache
  - 更像“单步 query 读大 cache”
  - 用 `flash_attn_with_kvcache`

- ## 支持 prefix cache
  
  if context.block_tables is not None:    # prefix cache  
  
      k, v = k_cache, v_cache
  
  这说明它支持前缀复用。  
  也就是已有前缀不重新算，直接从 cache 读。
  
  这对多请求共享系统 prompt、共享 prompt 前缀的场景非常重要。

- ## 变长序列支持
  
  prefill 用的是 `flash_attn_varlen_func`，结合：
  
  - `cu_seqlens_q`
  - `cu_seqlens_k`
  
  这说明它支持把不同长度请求压平混合计算，而不用 pad 成统一长度。
  
  这直接提升吞吐，减少无效计算。

- ## block table / slot mapping
  
  这两个东西说明 cache 不是按“每条序列一整段连续显存”简单管理，而是：
  
  - token -> slot：用 `slot_mapping`
  - sequence -> blocks：用 `block_tables`
  
  这正是分页式 KV Cache 管理的基础思路：
  
  - 更灵活
  - 更少碎片
  - 更适合请求动态加入、结束、复用

###### 4. 输出投影

```python
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

首先明确，atten输出的是[N, num_heads, head_dim]，然后通过[N, num_heads⋅head_dim]，表现为：

token0：[head_0的128个数 | head_1的128个数 | ... | head_27的128个数]

tokenn-1：[head_0的128个数 | head_1的128个数 | ... | head_27的128个数]

输出投影就是将泾渭分明的多头输出变为统一的hidden representation。

这里同样用到了tensor 并行的思想，通过将权重分配到不同gpu上。

**对比qkv的并行：**

qkv并行（ColumnParallelLinear）：

- 输入 `x` 是完整的
- 输出 `y` 是分片的
- 一般前向后不用立刻 all-reduce

o并行：

- 输入 `x` 是分片的
- 每张卡都算一个完整 shape 的部分输出
- 最后需要把这些部分输出相加，也就是 `all_reduce`

##### MLP

将attention得到的hidden_state(根据context得到的信息)，重新编码、筛选、放大、压缩，变成更有用的表示。

```python
class Qwen3MLP(nn.Module):
    """
    Qwen3模型的多层感知机(MLP)部分
    实现前馈网络，包含门控SiLU激活函数
    """

    def __init__(
        self,
        hidden_size: int,              # 隐藏层大小
        intermediate_size: int,         # 中间层大小
        hidden_act: str,                # 激活函数类型
    ) -> None:
        super().__init__()
        # 门控和上投影层，合并了两个投影操作
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # 两个相同大小的中间层投影
            bias=False,
        )
        # 下投影层，将中间层投影回隐藏层大小
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        # 激活函数必须是SiLU
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()  # SiLU和乘法激活函数

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: MLP输出
        """
        # 通过门控和上投影层
        gate_up = self.gate_up_proj(x)
        # 应用SiLU门控激活
        x = self.act_fn(gate_up)
        # 通过下投影层得到最终输出
        x = self.down_proj(x)
        return x
```

这是典型的gated MLP，先从输入 `x` 同时投影出两路中间表示，一路做门控 `gate`，一路做内容 `up`，然后用 `SiLU(gate)` 去调制 `up`，最后再投影回 hidden_size。

数学表达为：

$$
\begin{aligned}
u &= W_{\text{up}} x \\
g &= W_{\text{gate}} x \\
h &= \text{SiLU}(g) \odot u \\
y &= W_{\text{down}} h
\end{aligned}



$$

###### 1. gate_up_proj

```python
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

和之前的qkv投影计算一致，先将多个参数矩阵拼在一起，组成一个大矩阵，然后一次GEMM计算完成，这里也会用到TP的思想。

###### 2. down_proj

```python
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

和输出投影一样，这里是输入是只有各自的一部分，参数矩阵是相同的，每个设备计算在自己设备上的那一部分然后all_reduce.

###### 3. SiluAndMul

```python
class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
```

将x分为gate(x)，up(y)，然后通过silu函数将x变为缩放参数对y进行控制。最后输出的便是该层Decoder的hidden_state。

#### norm归一化

这一层与 Decoder中的RMSNorm一样，将输出进行一次归一化。

### ParallelLMHead （stage5）

```python
class ParallelLMHead(VocabParallelEmbedding):
    """
    并行语言模型头部
    继承自VocabParallelEmbedding，用于输出预测的词汇概率分布
    """

    def __init__(
        self,
        num_embeddings: int,  # 词汇表总大小
        embedding_dim: int,   # 嵌入维度
        bias: bool = False,   # 是否使用偏置（通常不使用）
    ):
        assert not bias  # 确保不使用偏置
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        x: 输入的隐藏状态
        """
        context = get_context()
        if context.is_prefill:
            # 在prefill阶段，只需要获取序列的最后一个位置的hidden state来预测下一个token
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # 使用线性层计算logits
        logits = F.linear(x, self.weight)

        if self.tp_size > 1:
            # 如果使用张量并行，则收集所有rank的logits
            # 创建列表存储所有rank的logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 将logits从所有rank聚集到rank 0
            dist.gather(logits, all_logits, 0)
            # 在rank 0上拼接所有logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        return logits
```

将model输出的hidden_state映射为词表维度的logits（还没有进行采样，softmax），这与VocabParallelEmbedding类似，每个设备计算logits的一部分，所以最后用gather拼成一个大向量：

例如：

- 总词表 151936
- TP=4

那么每张卡只算：

151936/4=37984

个词的 logit。

所以：

- rank0: `[B, 37984]`
- rank1: `[B, 37984]`
- rank2: `[B, 37984]`
- rank3: `[B, 37984]`

但如果你要真正做采样，最终需要完整词表 logits：[B, 151936]，那么将结果用gather拼起来即可。
