"""
embed_head.py
词汇并行嵌入层和平行语言模型头部的实现
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


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