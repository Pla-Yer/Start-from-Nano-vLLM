import os
import gc
import json
import time
import argparse
from typing import List, Dict

import torch
import torch.distributed as dist

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def cleanup():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def run_one_case(
    model_path: str,
    prompts: List[str],
    max_tokens: int,
    enforce_eager: bool,
    warmup_prompts: int = 2,
) -> Dict:
    llm = None
    try:
        llm = LLM(model=model_path, enforce_eager=enforce_eager)
        sp = SamplingParams(temperature=0.7, max_tokens=max_tokens)

        warmup_batch = prompts[:min(warmup_prompts, len(prompts))]
        _ = llm.generate(warmup_batch, sp, use_tqdm=False)

        # 如果是 cudagraph 模式，适当等一下，让 capture 更稳定
        if not enforce_eager:
            time.sleep(1)

        start = time.perf_counter()
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        elapsed = time.perf_counter() - start

        total_tokens = sum(len(o["token_ids"]) for o in outputs)

        result = {
            "mode": "eager" if enforce_eager else "cudagraph",
            "enforce_eager": enforce_eager,
            "total_time": elapsed,
            "throughput_tokens": total_tokens / elapsed if elapsed > 0 else 0.0,
            "throughput_requests": len(prompts) / elapsed if elapsed > 0 else 0.0,
            "num_requests": len(prompts),
            "num_tokens": total_tokens,
        }
        return result

    finally:
        try:
            if llm is not None and hasattr(llm, "exit"):
                llm.exit()
        except Exception:
            pass

        del llm
        cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["eager", "cudagraph"], required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing.",
        "Write a short poem.",
        "What is 2 + 2?",
        "Describe photosynthesis.",
        "Benefits of exercise?",
        "Tell me a joke.",
    ] * 4

    enforce_eager = (args.mode == "eager")
    result = run_one_case(
        model_path=args.model,
        prompts=test_prompts,
        max_tokens=args.max_tokens,
        enforce_eager=enforce_eager,
    )

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()