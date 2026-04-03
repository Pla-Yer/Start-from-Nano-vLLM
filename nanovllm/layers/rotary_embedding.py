"""
RoPE (Rotary Position Embedding) 旋转位置编码实现
该模块实现了旋转位置编码，它是一种有效的位置信息编码方式，
能够更好地捕捉序列中的相对位置和平移不变性。
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    对输入张量应用旋转位置编码
    
    Args:
        x: 输入张量，形状为 [..., rotary_dim]
        cos: 余弦值张量，形状为 [..., rotary_dim/2]
        sin: 正弦值张量，形状为 [..., rotary_dim/2]
    
    Returns:
        应用旋转编码后的张量，形状与输入x相同
    """
    # 将输入张量沿最后一维分成两半
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 应用旋转操作: [x1*cos - x2*sin, x2*cos + x1*sin]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 将结果拼接回原始形状，并转换回原始数据类型
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        use_torch_compile: bool = True,
    ) -> None:
        """
        初始化旋转位置编码
        
        Args:
            head_size: 查询/键头的维度
            rotary_dim: 用于旋转编码的维度（通常等于head_size）
            max_position_embeddings: 最大位置嵌入数
            base: 用于计算频率的基数，默认为10000
            use_torch_compile: 是否使用torch.compile优化
        """
        super().__init__()
        self.head_size = head_size
        self.use_torch_compile = use_torch_compile
        # 确保旋转编码的维度等于头的维度
        assert rotary_dim == head_size
        # 计算逆频率: 1/(base^(pos/dim))，其中pos是位置索引
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 创建位置张量
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 计算频率: 位置 × 逆频率
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 计算余弦和正弦值
        cos = freqs.cos()
        sin = freqs.sin()
        # 将余弦和正弦值连接起来，然后在第二个维度上增加一个维度，形成缓存
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # 注册为缓冲区，使其成为模型状态的一部分，但不参与梯度更新
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,  # 位置索引张量
        query: torch.Tensor,      # 查询张量
        key: torch.Tensor,        # 键张量
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，将旋转位置编码应用于查询和键张量
        
        Args:
            positions: 位置索引张量，形状为 [batch_size, seq_len]
            query: 查询张量，形状为 [batch_size, seq_len, num_heads, head_size]
            key: 键张量，形状为 [batch_size, seq_len, num_kv_heads, head_size]
        
        Returns:
            包含应用了旋转位置编码的查询和键张量的元组
        """
        # 根据位置索引从缓存中提取对应的cos和sin值
        cos_sin = self.cos_sin_cache[positions]
        # 将cos和sin分开
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 对查询和键应用旋转位置编码
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
    """
    获取旋转位置编码实例，使用LRU缓存避免重复创建
    
    Args:
        head_size: 查询/键头的维度
        rotary_dim: 用于旋转编码的维度
        max_position: 最大位置数
        base: 用于计算频率的基数
        rope_scaling: ROPE缩放配置（目前不支持）
    
    Returns:
        RotaryEmbedding实例
    """
    # 目前不支持ROPE缩放
    assert rope_scaling is None
    # 创建并返回旋转位置编码实例
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb