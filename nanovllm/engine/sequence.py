"""
Sequence类用于管理单个序列的状态和相关信息
"""
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举
    WAITING: 序列处于等待队列中，尚未开始运行
    RUNNING: 序列正在运行中，进行token生成
    FINISHED: 序列已完成，达到结束条件
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    序列类，管理单个输入序列的状态和相关信息
    """
    block_size = 256  # KV缓存块大小，每个块包含256个tokens
    counter = count()  # 用于生成唯一序列ID的计数器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化序列对象
        
        参数:
        - token_ids: 输入的token ID列表，包含原始prompt的token
        - sampling_params: 采样参数对象，控制生成行为（温度、最大token数等）
        
        属性:
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
        """
        self.seq_id = next(Sequence.counter)  # 分配唯一的序列ID
        self.status = SequenceStatus.WAITING  # 初始状态为等待
        self.token_ids = copy(token_ids)  # 复制输入的token ID列表
        self.last_token = token_ids[-1]  # 记录最后的token
        self.num_tokens = len(self.token_ids)  # 总token数量
        self.num_prompt_tokens = len(token_ids)  # 原始prompt的token数量
        self.num_cached_tokens = 0  # 已缓存token数量，初始为0
        self.prefill_chunk_size = 0  # 当前 step 计划处理的 prefill token 数
        self.block_table = []  # 块表，记录序列使用的物理块
        self.temperature = sampling_params.temperature  # 采样温度
        self.max_tokens = sampling_params.max_tokens  # 最大生成token数
        self.ignore_eos = sampling_params.ignore_eos  # 是否忽略EOS token

    def __len__(self):
        """
        返回序列的总长度（prompt + 已生成的token）
        """
        return self.num_tokens

    def __getitem__(self, key):
        """
        支持索引访问token_ids
        """
        return self.token_ids[key]

    @property
    def is_finished(self):
        """
        检查序列是否已完成
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """
        已生成的completion token数量
        等于总token数 - prompt token数
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_prompt_tokens_remaining(self):
        """
        还剩多少 prompt token 没有写入 KV cache
        """
        return max(0, self.num_prompt_tokens - self.num_cached_tokens)

    @property
    def is_prefill_finished(self):
        """
        prompt 是否已经全部完成 prefill
        """
        return self.num_cached_tokens >= self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """
        获取原始prompt的token ID列表
        """
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """
        获取已生成的completion token ID列表
        """
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """
        已缓存到KV缓存中的块数量
        根据已缓存的token数量计算
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        序列总共需要的块数量
        根据总token数计算
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        最后一个块中的token数量
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第i个块中的token
        
        参数:
        - i: 块索引
        
        返回:
        第i个块中的token列表
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        向序列中添加一个新的token
        
        参数:
        - token_id: 要添加的token ID
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        用于序列化的方法，返回对象状态
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.prefill_chunk_size, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """
        用于反序列化的设置状态方法
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.prefill_chunk_size, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
