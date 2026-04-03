"""
KV Cache prefix cache benchmark

修复后的公平对照版本：
- shared: 所有请求共享同一个长前缀
- unique: 每个请求拥有长度相同但内容不同的前缀

每个 case 在独立子进程中运行，避免 torch.distributed
重复 init_process_group。
"""

import json
import logging
import subprocess
import sys
import time
from typing import Dict, List, Tuple

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


def build_long_shared_prefix(repeat_tokens: int = 100) -> str:
    words = [f"policy{i}" for i in range(repeat_tokens)]
    text = (
        "system: You are a careful assistant. "
        "Please follow these instructions exactly. "
        + " ".join(words)
        + "\n"
        "user: "
    )
    return text


def build_long_unique_prefixes(num_prompts: int, repeat_tokens: int = 100) -> List[str]:
    prefixes = []
    for idx in range(num_prompts):
        words = [f"policy{idx}_{i}" for i in range(repeat_tokens)]
        text = (
            "system: You are a careful assistant. "
            "Please follow these instructions exactly. "
            + " ".join(words)
            + "\n"
            "user: "
        )
        prefixes.append(text)
    return prefixes


def build_prefix_benchmark_prompts(
    test_prompts: List[str],
    prefix_repeat_tokens: int = 100,
) -> Tuple[List[str], List[str]]:
    shared_prefix = build_long_shared_prefix(prefix_repeat_tokens)
    unique_prefixes = build_long_unique_prefixes(len(test_prompts), prefix_repeat_tokens)

    prompts_shared = [f"{shared_prefix}{p}" for p in test_prompts]
    prompts_unique = [f"{prefix}{p}" for prefix, p in zip(unique_prefixes, test_prompts)]
    return prompts_shared, prompts_unique


def benchmark_prefix_case(
    model_path: str,
    prompts: List[str],
    max_tokens: int = 8,
) -> Dict:
    llm = LLM(model=model_path)
    sp = SamplingParams(temperature=0.7, max_tokens=max_tokens)

    _ = llm.generate(prompts[:2], sp, use_tqdm=False)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    total_time = time.perf_counter() - start

    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)

    result = {
        "num_requests": len(prompts),
        "total_time": total_time,
        "throughput_tokens": total_output_tokens / total_time if total_time > 0 else 0.0,
        "throughput_requests": len(prompts) / total_time if total_time > 0 else 0.0,
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


def print_prefix_cache_results(results: Dict):
    print("\n" + "=" * 80)
    print("Prefix Cache Comparison")
    print("=" * 80)

    print("\nWith shared prefix:")
    r = results["with_prefix"]
    print(f"  Time: {r['total_time']:.3f}s")
    print(f"  Throughput (tok/s): {r['throughput_tokens']:.2f}")
    print(f"  Throughput (req/s): {r['throughput_requests']:.2f}")

    print("\nWithout shared prefix:")
    r = results["without_prefix"]
    print(f"  Time: {r['total_time']:.3f}s")
    print(f"  Throughput (tok/s): {r['throughput_tokens']:.2f}")
    print(f"  Throughput (req/s): {r['throughput_requests']:.2f}")

    imp = results["improvement"]
    print("\nImprovement:")
    print(f"  Time reduction: {imp['time_reduction_percent']:.1f}%")
    print(f"  Throughput increase: {imp['throughput_increase_percent']:.1f}%")

    print("=" * 80)


def worker_main():
    """
    python benchmark_prefix.py worker <mode> <model_path> <prefix_repeat_tokens> <max_tokens>
    mode: prefix_shared | prefix_unique
    """
    mode = sys.argv[2]
    model_path = sys.argv[3]
    prefix_repeat_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    test_prompts = build_test_prompts(num_repeats=4)
    prompts_shared, prompts_unique = build_prefix_benchmark_prompts(
        test_prompts=test_prompts,
        prefix_repeat_tokens=prefix_repeat_tokens,
    )

    prompts = prompts_shared if mode == "prefix_shared" else prompts_unique
    result = benchmark_prefix_case(
        model_path=model_path,
        prompts=prompts,
        max_tokens=max_tokens,
    )
    print(json.dumps(result, ensure_ascii=False))


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6b-Instruct"
    script_path = __file__

    prefix_repeat_tokens = 100
    max_tokens = 8

    print("\n### Prefix Cache Benchmark ###")

    prefix_shared = run_subprocess_case(
        script_path,
        ["worker", "prefix_shared", model_path, str(prefix_repeat_tokens), str(max_tokens)],
    )
    prefix_unique = run_subprocess_case(
        script_path,
        ["worker", "prefix_unique", model_path, str(prefix_repeat_tokens), str(max_tokens)],
    )

    results = {
        "with_prefix": prefix_shared,
        "without_prefix": prefix_unique,
        "improvement": {
            "time_reduction_percent": (
                (prefix_unique["total_time"] - prefix_shared["total_time"])
                / prefix_unique["total_time"] * 100
                if prefix_unique["total_time"] > 0 else 0.0
            ),
            "throughput_increase_percent": (
                (prefix_shared["throughput_tokens"] - prefix_unique["throughput_tokens"])
                / prefix_unique["throughput_tokens"] * 100
                if prefix_unique["throughput_tokens"] > 0 else 0.0
            ),
        }
    }

    print_prefix_cache_results(results)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_main()
    else:
        main()