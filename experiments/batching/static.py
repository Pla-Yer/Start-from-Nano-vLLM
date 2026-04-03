"""
Static Batching 实验
固定 batch size 进行推理，对比不同 batch size 的性能
"""
import time
import asyncio
from typing import List
from dataclasses import dataclass
import logging

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    batch_size: int
    num_requests: int
    total_latency: float
    avg_latency: float
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float


async def run_static_batching(
    llm: LLM,
    prompts: List[str],
    batch_size: int,
    max_tokens: int = 64,
    temperature: float = 0.7,
) -> BenchmarkResult:
    """
    运行固定 batch size 的推理

    Args:
        llm: LLM 实例
        prompts: prompt 列表
        batch_size: 固定 batch size
        max_tokens: 最大生成 token 数
        temperature: 采样温度

    Returns:
        BenchmarkResult: 测试结果
    """
    num_requests = len(prompts)
    all_latencies = []

    # 分批处理
    for i in range(0, num_requests, batch_size):
        batch_prompts = prompts[i:i + batch_size]
        if not batch_prompts:
            continue

        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # 计时
        start_time = time.perf_counter()
        outputs = llm.generate(batch_prompts, sp, use_tqdm=False)
        end_time = time.perf_counter()

        batch_latency = end_time - start_time
        all_latencies.append(batch_latency)

        logger.info(
            f"Batch {i // batch_size + 1}: "
            f"{len(batch_prompts)} requests in {batch_latency:.3f}s"
        )

    total_latency = sum(all_latencies)
    avg_latency = total_latency / len(all_latencies) if all_latencies else 0

    # 计算吞吐量
    total_tokens = sum(
        len(output["token_ids"])
        for output in outputs
        for _ in [1]  # iterate once
    )

    # 重新运行获取总 token 数（简化处理）
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts[:batch_size], sp, use_tqdm=False)
    total_tokens = sum(len(o["token_ids"]) for o in outputs) * (num_requests // batch_size)

    throughput_tokens = total_tokens / total_latency if total_latency > 0 else 0
    throughput_requests = num_requests / total_latency if total_latency > 0 else 0

    return BenchmarkResult(
        batch_size=batch_size,
        num_requests=num_requests,
        total_latency=total_latency,
        avg_latency=avg_latency,
        throughput_tokens_per_sec=throughput_tokens,
        throughput_requests_per_sec=throughput_requests,
    )


async def compare_batch_sizes(
    model_path: str,
    prompts: List[str],
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64, 128],
    max_tokens: int = 256,
) -> List[BenchmarkResult]:
    """
    对比不同 batch size 的性能

    Args:
        model_path: 模型路径
        prompts: 测试 prompts
        batch_sizes: 要测试的 batch sizes
        max_tokens: 最大生成 token 数

    Returns:
        各 batch size 的测试结果
    """
    logger.info(f"Loading model: {model_path}")
    llm = LLM(model=model_path)

    results = []
    for bs in batch_sizes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing batch_size={bs}")
        logger.info(f"{'='*50}")

        result = await run_static_batching(
            llm,
            prompts,
            batch_size=bs,
            max_tokens=max_tokens,
        )

        results.append(result)

        logger.info(f"Results for batch_size={bs}:")
        logger.info(f"  Total latency: {result.total_latency:.3f}s")
        logger.info(f"  Avg latency: {result.avg_latency:.3f}s")
        logger.info(f"  Throughput (tokens/s): {result.throughput_tokens_per_sec:.2f}")
        logger.info(f"  Throughput (req/s): {result.throughput_requests_per_sec:.2f}")

    return results


def print_comparison(results: List[BenchmarkResult]):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("Static Batching Comparison Results")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Requests':<10} {'Total Time':<12} {'Avg Latency':<12} {'Tok/s':<12} {'Req/s':<12}")
    print("-" * 80)

    for r in results:
        print(
            f"{r.batch_size:<12} "
            f"{r.num_requests:<10} "
            f"{r.total_latency:<12.3f} "
            f"{r.avg_latency:<12.3f} "
            f"{r.throughput_tokens_per_sec:<12.2f} "
            f"{r.throughput_requests_per_sec:<12.2f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    import sys

    # 示例 prompts
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about spring.",
        "What is 2 + 2?",
        "Describe the process of photosynthesis.",
        "What are the benefits of exercise?",
        "Tell me a joke.",
    ] * 16  # 重复以获得更多请求

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6b-Instruct"

    results = asyncio.run(compare_batch_sizes(model_path, test_prompts))
    print_comparison(results)