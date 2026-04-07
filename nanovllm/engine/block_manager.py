from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

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
            
            # cache miss 的块要等真正完成 prefill 后再登记哈希，避免 chunked prefill
            # 下其他请求误复用尚未写完 KV 的 block。
            if h != -1 and not cache_miss:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            # 将块ID添加到序列的块表中
            seq.block_table.append(block_id)

    def cache_full_blocks(self, seq: Sequence, prev_num_cached_tokens: int, num_cached_tokens: int):
        start_block = prev_num_cached_tokens // self.block_size
        end_block = num_cached_tokens // self.block_size
        for i in range(start_block, end_block):
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            if block.hash != -1:
                continue
            token_ids = seq.block(i)
            prefix = self.blocks[seq.block_table[i-1]].hash if i > 0 else -1
            h = self.compute_hash(token_ids, prefix)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id

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

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
