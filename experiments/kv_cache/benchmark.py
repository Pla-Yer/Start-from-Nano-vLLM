"""
KV Cache block size benchmark

每个 block size 在独立子进程中运行，避免 torch.distributed
重复 init_process_group。
"""

import json
import logging
import subprocess
import sys
import time
from typing import Dict, List

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_test_prompts(num_repeats: int = 4) -> List[str]:
    base = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about spring.",
        "What is 2 + 2?",
        "Describe photosynthesis briefly.",
        "What are the benefits of exercise?",
        "Tell me a joke.",
    ]
    return base * num_repeats


def benchmark_one_block_size(
    model_path: str,
    prompts: List[str],
    block_size: int,
    max_tokens: int = 64,
) -> Dict:
    logger.info(f"Testing block_size={block_size}")

    llm = LLM(
        model=model_path,
        kvcache_block_size=block_size,
    )
    sp = SamplingParams(temperature=0.7, max_tokens=max_tokens)

    _ = llm.generate(prompts[:2], sp, use_tqdm=False)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    total_time = time.perf_counter() - start

    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)

    result = {
        "block_size": block_size,
        "total_time": total_time,
        "throughput_tokens": total_output_tokens / total_time if total_time > 0 else 0.0,
        "throughput_requests": len(prompts) / total_time if total_time > 0 else 0.0,
        "avg_latency": total_time / len(prompts) if prompts else 0.0,
        "total_output_tokens": total_output_tokens,
    }

    del llm
    return result


def run_subprocess_case(script_path: str, args: List[str]) -> Dict:
    cmd = [sys.executable, script_path] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if proc.returncode != 0:
        print("Subprocess failed.")
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Subprocess failed with code {proc.returncode}")

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    raise RuntimeError(f"Cannot parse subprocess output as JSON:\n{proc.stdout}")


def print_block_size_results(results: Dict):
    print("\n" + "=" * 80)
    print("KV Cache Block Size Comparison")
    print("=" * 80)
    print(f"{'Block Size':<12} {'Time':<12} {'Tok/s':<12} {'Req/s':<12} {'Avg Latency':<12}")
    print("-" * 80)

    for bs, r in results.items():
        print(
            f"{bs:<12} "
            f"{r['total_time']:<12.3f} "
            f"{r['throughput_tokens']:<12.2f} "
            f"{r['throughput_requests']:<12.2f} "
            f"{r['avg_latency']:<12.3f}"
        )

    print("=" * 80)


def worker_main():
    """
    python benchmark_block.py worker <model_path> <block_size> <max_tokens>
    """
    model_path = sys.argv[2]
    block_size = int(sys.argv[3])
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 64

    prompts = build_test_prompts(num_repeats=4)
    result = benchmark_one_block_size(
        model_path=model_path,
        prompts=prompts,
        block_size=block_size,
        max_tokens=max_tokens,
    )
    print(json.dumps(result, ensure_ascii=False))


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6b-Instruct"
    script_path = __file__

    block_sizes = [256, 512, 1024]
    max_tokens = 64

    print("\n### Block Size Benchmark ###")
    results = {}
    for bs in block_sizes:
        result = run_subprocess_case(
            script_path,
            ["worker", model_path, str(bs), str(max_tokens)],
        )
        results[bs] = result

    print_block_size_results(results)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_main()
    else:
        main()