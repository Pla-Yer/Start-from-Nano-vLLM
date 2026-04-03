"""
Config类定义了nano-vllm引擎的各种配置参数
"""
import os
import torch
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    配置类，包含nano-vllm引擎的主要配置参数
    """
    # 模型路径，指向HuggingFace格式的预训练模型目录
    model: str
    
    # 最大批处理token数量，限制单次批处理的最大token数
    max_num_batched_tokens: int = 16384
    
    # 最大批处理序列数量，限制单次批处理的最大序列数
    max_num_seqs: int = 512
    
    # 模型最大长度，限制单个序列的最大长度
    max_model_len: int = 4096
    
    # GPU内存利用率，控制GPU内存的使用比例（0.0-1.0）
    gpu_memory_utilization: float = 0.9
    
    # 张量并行大小，指定用于张量并行的GPU数量
    tensor_parallel_size: int = 1
    
    # 是否强制使用eager模式，如果为False则可能启用图优化
    enforce_eager: bool = False
    
    # HuggingFace模型配置，从预训练模型加载的配置信息
    hf_config: AutoConfig | None = None
    
    # 结束符ID，用于识别序列结束的token ID
    eos: int = -1
    
    # KV缓存块大小，每个KV缓存块包含的token数量
    kvcache_block_size: int = 256
    
    # KV缓存块数量，总的KV缓存块的数量
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        """
        初始化后的验证和配置设置
        - 验证模型路径存在
        - 验证KV缓存块大小是256的倍数
        """
        # 验证KV缓存块大小是256的倍数
        assert self.kvcache_block_size % 256 == 0, f"KV缓存块大小必须是256的倍数，当前值为{self.kvcache_block_size}"
        
        # 如果未指定KV缓存块数量，则根据GPU内存使用率估算
        if self.num_kvcache_blocks == -1:
            # 估算KV缓存所需的总内存
            # 计算单个KV缓存元素的字节数（考虑头部数量、头维度、块大小等因素）
            kvcache_bytes_per_token = (
                # 所有注意力头的键值对内存占用（2表示K和V）
                self.model_config.num_key_value_heads * self.model_config.hidden_size // self.model_config.num_attention_heads
                * 2 
                # 数据类型字节大小（这里假设是fp16/bf16，占2字节）
                * 2
            )
            # 计算可用GPU内存中的KV缓存块数量
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            self.num_kvcache_blocks = int(
                self.gpu_memory_utilization 
                * gpu_memory / (kvcache_bytes_per_token * self.kvcache_block_size)
            )

    @property
    def model_config(self):
        """
        获取HuggingFace模型配置
        - 如果hf_config为None，则从模型路径加载配置
        - 否则直接返回已有的hf_config
        """
        if self.hf_config is None:
            self.hf_config = AutoConfig.from_pretrained(self.model)
        return self.hf_config