import asyncio
import gc
import logging
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Request:
    req_id: int
    prompt: str
    prompt_type: str
    arrival_time: float
    max_tokens: int = 64
    temperature: float = 0.7
    seq_id: Optional[int] = None
    seq: Optional[Any] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def latency(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time


@dataclass
class ExperimentResult:
    strategy: str
    total_time: float
    throughput_req_s: float
    avg_latency: float
    p95_latency: float
    avg_ttft: float
    p95_ttft: float
    short_avg_latency: float
    short_p95_latency: float
    short_avg_ttft: float
    short_p95_ttft: float
    long_avg_latency: float


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((p / 100.0) * (len(values) - 1)))
    return values[idx]


def make_prompt(token_budget: int, word: str) -> str:
    return " ".join([word] * token_budget)


def build_workload(
    num_long_prompts: int = 2,
    num_short_prompts: int = 12,
    long_prompt_words: int = 2048,
    short_prompt_words: int = 32,
    arrival_gap_s: float = 0.03,
) -> List[Request]:
    requests: List[Request] = []
    req_id = 0

    for i in range(num_long_prompts):
        requests.append(
            Request(
                req_id=req_id,
                prompt=make_prompt(long_prompt_words, f"long{i}"),
                prompt_type="long",
                arrival_time=0.0,
                max_tokens=64,
            )
        )
        req_id += 1

    for i in range(num_short_prompts):
        arrival = arrival_gap_s * (i + 1) + random.uniform(0.0, arrival_gap_s * 0.2)
        requests.append(
            Request(
                req_id=req_id,
                prompt=make_prompt(short_prompt_words, f"short{i}"),
                prompt_type="short",
                arrival_time=arrival,
                max_tokens=48,
            )
        )
        req_id += 1

    requests.sort(key=lambda req: req.arrival_time)
    return requests


async def run_workload(llm: LLM, requests: List[Request]) -> ExperimentResult:
    start_wall = time.perf_counter()
    active: Dict[int, Request] = {}
    finished: List[Request] = []
    i = 0

    while i < len(requests) or active or not llm.is_finished():
        now = time.perf_counter() - start_wall

        while i < len(requests) and requests[i].arrival_time <= now:
            req = requests[i]
            sp = SamplingParams(temperature=req.temperature, max_tokens=req.max_tokens)
            llm.add_request(req.prompt, sp)
            req.seq = llm.scheduler.waiting[-1]
            req.seq_id = req.seq.seq_id
            active[req.seq_id] = req
            i += 1

        if active or not llm.is_finished():
            outputs, _ = llm.step()
            now = time.perf_counter() - start_wall
            for req in active.values():
                if req.first_token_time is None and req.seq is not None and req.seq.num_completion_tokens > 0:
                    req.first_token_time = now
            for seq_id, _token_ids in outputs:
                if seq_id not in active:
                    continue
                req = active.pop(seq_id)
                if req.first_token_time is None:
                    req.first_token_time = now
                req.finish_time = now
                finished.append(req)
        else:
            await asyncio.sleep(0.001)

        await asyncio.sleep(0)

    total_time = time.perf_counter() - start_wall
    latencies = [req.latency for req in finished if req.latency is not None]
    ttfts = [req.ttft for req in finished if req.ttft is not None]
    short_latencies = [req.latency for req in finished if req.prompt_type == "short" and req.latency is not None]
    short_ttfts = [req.ttft for req in finished if req.prompt_type == "short" and req.ttft is not None]
    long_latencies = [req.latency for req in finished if req.prompt_type == "long" and req.latency is not None]
    strategy = "chunked_prefill" if llm.scheduler.enable_chunked_prefill else "baseline"
    return ExperimentResult(
        strategy=strategy,
        total_time=total_time,
        throughput_req_s=len(finished) / total_time if total_time > 0 else 0.0,
        avg_latency=statistics.mean(latencies) if latencies else 0.0,
        p95_latency=percentile(latencies, 95),
        avg_ttft=statistics.mean(ttfts) if ttfts else 0.0,
        p95_ttft=percentile(ttfts, 95),
        short_avg_latency=statistics.mean(short_latencies) if short_latencies else 0.0,
        short_p95_latency=percentile(short_latencies, 95),
        short_avg_ttft=statistics.mean(short_ttfts) if short_ttfts else 0.0,
        short_p95_ttft=percentile(short_ttfts, 95),
        long_avg_latency=statistics.mean(long_latencies) if long_latencies else 0.0,
    )


async def compare_chunked_prefill(
    model_path: str,
    prefill_chunk_sizes: List[int],
    enforce_eager: bool = True,
):
    workload = build_workload()
    chunk_sizes = []
    for size in prefill_chunk_sizes:
        if size > 0 and size not in chunk_sizes:
            chunk_sizes.append(size)
    configs = [dict(enable_chunked_prefill=False, enforce_eager=enforce_eager)]
    configs.extend(
        dict(enable_chunked_prefill=True, prefill_chunk_size=size, enforce_eager=enforce_eager)
        for size in chunk_sizes
    )
    results: List[ExperimentResult] = []

    for config in configs:
        logger.info("Loading model with config=%s", config)
        llm = LLM(model=model_path, **config)
        warmup_sp = SamplingParams(temperature=0.1, max_tokens=8)
        llm.generate(["warmup"], warmup_sp, use_tqdm=False)

        requests = [
            Request(
                req_id=req.req_id,
                prompt=req.prompt,
                prompt_type=req.prompt_type,
                arrival_time=req.arrival_time,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            for req in workload
        ]
        result = await run_workload(llm, requests)
        if config["enable_chunked_prefill"]:
            result.strategy = f"chunk_{config['prefill_chunk_size']}"
        results.append(result)
        llm.exit()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        await asyncio.sleep(0.1)

    print(
        f"{'Strategy':<18} {'Total(s)':>10} {'Req/s':>10} "
        f"{'Avg TTFT':>10} {'P95 TTFT':>10} "
        f"{'Short TTFT':>11} {'Short P95':>10} "
        f"{'Avg Lat':>10} {'Short Lat':>10}"
    )
    for result in results:
        print(
            f"{result.strategy:<18} {result.total_time:>10.3f} {result.throughput_req_s:>10.2f} "
            f"{result.avg_ttft:>10.3f} {result.p95_ttft:>10.3f} "
            f"{result.short_avg_ttft:>11.3f} {result.short_p95_ttft:>10.3f} "
            f"{result.avg_latency:>10.3f} {result.short_avg_latency:>10.3f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare baseline scheduling vs chunked prefill sweep.")
    parser.add_argument("--model", type=str,  default="/home/ttt/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B", help="Model path.")
    parser.add_argument(
        "--prefill-chunk-sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Prompt tokens per prefill chunk. Supports multiple values for automatic sweep.",
    )
    parser.add_argument("--allow-cudagraph", action="store_false", dest="enforce_eager", help="Enable CUDA graph during the experiment.")
    parser.set_defaults(enforce_eager=True)
    args = parser.parse_args()
    asyncio.run(compare_chunked_prefill(args.model, args.prefill_chunk_sizes, args.enforce_eager))
