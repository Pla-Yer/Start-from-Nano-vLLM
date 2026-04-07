from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.prefill_chunk_size = config.prefill_chunk_size
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._schedule_prefill_next = True

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度序列进行处理，实现预填充(prefill)和解码(decode)的交替调度
        
        Returns:
            tuple[list[Sequence], bool]: 返回被调度的序列列表和一个布尔值，
            布尔值表示是否进行了预填充(True表示预填充，False表示解码)
        """
        # 当启用分块预填充且等待队列和运行队列都不为空时
        # 这种情况下实现了prefill和decode的交替调度机制
        if self.enable_chunked_prefill and self.waiting and self.running:
            # 根据_schedule_prefill_next标志决定本次调度优先处理哪种类型的任务
            if self._schedule_prefill_next:
                # 尝试调度预填充任务
                scheduled_seqs = self._schedule_prefill()
                if scheduled_seqs:
                    # 成功调度了预填充任务，将标志设为False，
                    # 下次将优先调度解码任务
                    self._schedule_prefill_next = False
                    return scheduled_seqs, True  # True表示是预填充任务
                # 如果没有成功调度预填充任务（等待队列为空等），则调度解码任务
                scheduled_seqs = self._schedule_decode()
                # 因为本次调度的是解码任务，将标志设为True，
                # 下次将优先调度预填充任务
                self._schedule_prefill_next = True
                return scheduled_seqs, False  # False表示是解码任务

            # 执行解码调度（当_schedule_prefill_next为False时）
            scheduled_seqs = self._schedule_decode()
            if scheduled_seqs:
                # 成功调度了解码任务，将标志设为True，
                # 下次将优先调度预填充任务
                self._schedule_prefill_next = True
                return scheduled_seqs, False  # False表示是解码任务
            # 如果没有成功调度解码任务，则尝试调度预填充任务
            scheduled_seqs = self._schedule_prefill()
            # 因为本次调度的是预填充任务，将标志设为False，
            # 下次将优先调度解码任务
            self._schedule_prefill_next = False
            return scheduled_seqs, True  # True表示是预填充任务

        # 如果未启用分块预填充或某个队列为空，则按顺序调度：
        # 优先尝试调度预填充任务（从等待队列中取出序列进行预填充）
        scheduled_seqs = self._schedule_prefill()
        if scheduled_seqs:
            self._schedule_prefill_next = False
            return scheduled_seqs, True  # True表示是预填充任务

        # 如果没有预填充任务可调度，则调度解码任务（运行队列中的序列继续生成token）
        scheduled_seqs = self._schedule_decode()
        # 调度完解码任务后，下次将优先调度预填充任务
        self._schedule_prefill_next = True
        return scheduled_seqs, False  # False表示是解码任务

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        token_budget = self.prefill_chunk_size if self.enable_chunked_prefill else self.max_num_batched_tokens
        while self.waiting and num_seqs < self.max_num_seqs and num_batched_tokens < token_budget:
            seq = self.waiting.popleft()
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    self.waiting.appendleft(seq)
                    break
                self.block_manager.allocate(seq)

            if seq.is_prefill_finished:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                continue

            chunk_size = seq.num_prompt_tokens_remaining if not self.enable_chunked_prefill else min(
                seq.num_prompt_tokens_remaining,
                token_budget - num_batched_tokens,
            )
            if chunk_size <= 0:
                self.waiting.appendleft(seq)
                break
            num_seqs += 1
            num_batched_tokens += chunk_size
            seq.prefill_chunk_size = chunk_size
            if chunk_size == seq.num_prompt_tokens_remaining:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                # 未完成的 prefill 请求放回 waiting 队尾，给其他请求插入机会。
                self.waiting.append(seq)
            scheduled_seqs.append(seq)
            if self.enable_chunked_prefill and num_batched_tokens >= token_budget:
                break
        return scheduled_seqs

    def _schedule_decode(self) -> list[Sequence]:
        scheduled_seqs = []
        num_seqs = 0
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
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.prefill_chunk_size = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                prev_num_cached_tokens = seq.num_cached_tokens
                seq.num_cached_tokens += seq.prefill_chunk_size
                self.block_manager.cache_full_blocks(seq, prev_num_cached_tokens, seq.num_cached_tokens)
                seq.prefill_chunk_size = 0
                if not seq.is_prefill_finished:
                    continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
