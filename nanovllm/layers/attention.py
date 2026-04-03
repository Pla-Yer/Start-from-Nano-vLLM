"""
注意力机制实现
该模块实现了带有KV缓存的Flash Attention，支持prefill和decode两种推理阶段
"""

import torch
from torch import nn
import triton
import triton.language as tl

# 导入flash attention相关函数，用于高效计算注意力
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,           # 键张量指针
    key_stride,        # 键张量步长
    value_ptr,         # 值张量指针
    value_stride,      # 值张量步长
    k_cache_ptr,       # 键缓存张量指针
    v_cache_ptr,       # 值缓存张量指针
    slot_mapping_ptr,  # 槽位映射指针，指定每个token存储到哪个槽位
    D: tl.constexpr,   # 头部维度
):
    """
    Triton核函数，将键和值存储到KV缓存中
    使用Triton JIT编译以提高GPU执行效率
    """
    # 获取当前程序ID（即当前处理的token索引）
    idx = tl.program_id(0)
    # 获取该token应该存储到缓存中的槽位
    slot = tl.load(slot_mapping_ptr + idx)
    # 如果槽位为-1，表示不需要存储，直接返回
    if slot == -1: 
        return
    
    # 计算键张量中当前token的偏移量
    key_offsets = idx * key_stride + tl.arange(0, D)
    # 计算值张量中当前token的偏移量
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    # 从输入张量中加载键和值
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # 计算缓存中的偏移量
    cache_offsets = slot * D + tl.arange(0, D)
    
    # 将键和值存储到对应槽位的缓存中
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    将键和值张量存储到KV缓存中
    这是一个包装函数，用于调用Triton内核
    
    Args:
        key: 键张量，形状为[N, num_heads, head_dim]
        value: 值张量，形状为[N, num_heads, head_dim]
        k_cache: 键缓存张量
        v_cache: 值缓存张量
        slot_mapping: 槽位映射，指定每个位置的token存储到缓存中的哪个槽位
    """
    N, num_heads, head_dim = key.shape  # 获取输入张量的形状
    D = num_heads * head_dim            # 计算总的头部维度
    # 确保张量的内存布局符合要求（最后一维连续）
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    # 确保第二维的步长等于头维度
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # 确保缓存张量的第二维步长等于总头维度
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    # 确保槽位映射的数量与序列长度一致
    assert slot_mapping.numel() == N
    
    # 启动Triton内核，每个token由一个程序处理
    store_kvcache_kernel[(N,)](
        key, key.stride(0),           # 键张量及其第一维步长
        value, value.stride(0),       # 值张量及其第一维步长
        k_cache, v_cache,             # 键缓存和值缓存
        slot_mapping,                 # 槽位映射
        D                             # 总头部维度
    )


class Attention(nn.Module):
    """
    注意力层实现
    支持分组查询注意力（GQA）和多查询注意力（MQA）
    """

    def __init__(
        self,
        num_heads,      # 查询头的数量
        head_dim,       # 每个头的维度
        scale,          # 注意力分数的缩放因子
        num_kv_heads,   # 键和值的头数量
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        
        # 初始化KV缓存为空张量
        # 在实际推理过程中，这些缓存会被设置为模型的全局KV缓存
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        注意力层的前向传播
        
        Args:
            q: 查询张量
            k: 键张量
            v: 值张量
            
        Returns:
            注意力输出张量
        """
        # 获取当前上下文，包含推理所需的各种元数据
        context = get_context()
        # 获取KV缓存引用
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 如果缓存不为空，则将当前的k和v存储到缓存中
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # 根据当前推理阶段选择不同的注意力计算方式
        if context.is_prefill:
            # Prefill阶段：处理新输入的prompt
            if context.block_tables is not None:    # 使用前缀缓存的情况
                k, v = k_cache, v_cache
            
            # 使用变长Flash Attention计算输出
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,   # 查询的最大序列长度
                cu_seqlens_q=context.cu_seqlens_q,   # 查询序列长度的累积和
                max_seqlen_k=context.max_seqlen_k,   # 键的最大序列长度
                cu_seqlens_k=context.cu_seqlens_k,   # 键序列长度的累积和
                softmax_scale=self.scale,            # softmax缩放因子
                causal=True,                         # 使用因果掩码，防止未来信息泄露
                block_table=context.block_tables     # 块表，用于间接访问缓存
            )
        else:    # Decode阶段：逐个生成token
            # 使用带KV缓存的Flash Attention进行解码
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),                     # 查询张量增加一个维度
                k_cache,                            # 键缓存
                v_cache,                            # 值缓存
                cache_seqlens=context.context_lens, # 缓存中的序列长度
                block_table=context.block_tables,   # 块表
                softmax_scale=self.scale,           # softmax缩放因子
                causal=True                         # 因果掩码
            )
            
        return o