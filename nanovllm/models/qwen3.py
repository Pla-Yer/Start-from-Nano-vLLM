"""
Qwen3模型的实现
包含注意力机制、前馈网络、解码器层等组件
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


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


class Qwen3Model(nn.Module):
    """
    Qwen3模型主干网络
    包含词嵌入、多个解码器层和最终的归一化层
    """

    def __init__(
        self,
        config: Qwen3Config,  # 模型配置
    ) -> None:
        super().__init__()
        # 词汇表并行嵌入层，将token ID转换为向量
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 创建多个解码器层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终的归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,   # 输入token ID张量
        positions: torch.Tensor,   # 位置索引张量
    ) -> torch.Tensor:
        """
        前向传播
        :param input_ids: 输入token ID
        :param positions: 位置索引
        :return: 模型输出的隐藏状态
        """
        # 将输入ID转换为嵌入向量
        hidden_states = self.embed_tokens(input_ids)
        residual = None  # 初始时没有残差
        
        # 依次通过每一层解码器
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 对最终输出进行归一化
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    用于因果语言建模的Qwen3模型
    包含完整的模型结构和语言模型头
    """
    
    # # 定义打包模块的映射关系，用于处理模型权重的加载
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