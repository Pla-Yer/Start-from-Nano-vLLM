import time
import asyncio
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Request:
    req_id: int
    prompt: str
    arrival_time: float
    temperature: float = 0.7
    max_tokens: int = 64

    # 统计信息
    submit_time: Optional[float] = None
    finish_time: Optional[float] = None
    seq_id: Optional[int] = None
    output_token_ids: List[int] = field(default_factory=list)

    @property
    def latency(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time




@dataclass
class BenchmarkResult:
    strategy: str
    num_requests: int
    total_time: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    throughput_tokens: float
    throughput_requests: float

    # 新增：实际 batch 统计
    avg_actual_batch_size: float = 0.0
    p50_actual_batch_size: float = 0.0
    p95_actual_batch_size: float = 0.0
    actual_batch_sizes: List[int] = field(default_factory=list)
    actual_batch_histogram: Dict[int, int] = field(default_factory=dict)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((p / 100.0) * (len(values) - 1)))
    return values[idx]


def build_requests(
    prompts: List[str],
    arrival_rate: float = 5.0,
    jitter: float = 0.0,
    temperature: float = 0.7,
    max_tokens: int = 64,
) -> List[Request]:
    requests = []
    for i, prompt in enumerate(prompts):
        t = i / arrival_rate
        if jitter > 0:
            t += random.uniform(-jitter, jitter)
            t = max(0.0, t)
        requests.append(
            Request(
                req_id=i,
                prompt=prompt,
                arrival_time=t,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    requests.sort(key=lambda r: r.arrival_time)
    return requests


async def run_static_batching(
    llm: LLM,
    requests: List[Request],
    batch_size: int,
    flush_timeout: float = 0.2,
) -> BenchmarkResult:
    """
    static batching:
    - 请求按 arrival_time 到达
    - 先放入 pending
    - 满 batch 或等待超时再一次性送入引擎
    - 送入后，等这批全部完成，再送下一批
    """
    start_wall = time.perf_counter()
    pending: List[Request] = []
    done: List[Request] = []
    total_tokens = 0
    i = 0
    n = len(requests)

    # 新增：记录每次真正发出去的 batch 大小
    actual_batch_sizes: List[int] = []

    while i < n or pending:
        now = time.perf_counter() - start_wall

        # 收集已到达请求
        while i < n and requests[i].arrival_time <= now:
            pending.append(requests[i])
            i += 1

        should_flush = False
        flush_reason = None

        if len(pending) >= batch_size:
            should_flush = True
            flush_reason = "full"
        elif pending:
            oldest_wait = now - pending[0].arrival_time
            if oldest_wait >= flush_timeout:
                should_flush = True
                flush_reason = "timeout"

        if i >= n and pending:
            should_flush = True
            flush_reason = "drain"

        if not should_flush:
            await asyncio.sleep(0.001)
            continue

        # 取一批
        batch = pending[:batch_size]
        pending = pending[batch_size:]

        # 记录这次实际 batch size
        actual_bs = len(batch)
        actual_batch_sizes.append(actual_bs)

        logger.info(
            f"[static_{batch_size}] flush_reason={flush_reason}, "
            f"actual_batch_size={actual_bs}, pending_left={len(pending)}, "
            f"arrived={i}/{n}, t={time.perf_counter() - start_wall:.3f}s"
        )

        # 逐个 add_request，并记录 seq_id
        seqid_to_req: Dict[int, Request] = {}
        for req in batch:
            sp = SamplingParams(
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            req.submit_time = time.perf_counter() - start_wall
            req.seq_id = llm.add_request(req.prompt, sp)
            seqid_to_req[req.seq_id] = req

        # 跑到这一批全部完成
        remaining = set(seqid_to_req.keys())
        while remaining:
            outputs, _ = llm.step()
            now_finish = time.perf_counter() - start_wall

            for seq_id, token_ids in outputs:
                if seq_id in remaining:
                    req = seqid_to_req[seq_id]
                    req.output_token_ids = token_ids
                    req.finish_time = now_finish
                    total_tokens += len(token_ids)
                    done.append(req)
                    remaining.remove(seq_id)

        await asyncio.sleep(0)

    total_time = time.perf_counter() - start_wall
    latencies = [r.latency for r in done if r.latency is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # 新增：batch size 统计
    avg_actual_bs = (
        sum(actual_batch_sizes) / len(actual_batch_sizes)
        if actual_batch_sizes else 0.0
    )
    p50_actual_bs = percentile(actual_batch_sizes, 50) if actual_batch_sizes else 0.0
    p95_actual_bs = percentile(actual_batch_sizes, 95) if actual_batch_sizes else 0.0
    hist = dict(sorted(Counter(actual_batch_sizes).items()))

    return BenchmarkResult(
        strategy=f"static_{batch_size}",
        num_requests=len(done),
        total_time=total_time,
        avg_latency=avg_latency,
        p50_latency=percentile(latencies, 50),
        p95_latency=percentile(latencies, 95),
        throughput_tokens=total_tokens / total_time if total_time > 0 else 0.0,
        throughput_requests=len(done) / total_time if total_time > 0 else 0.0,
        avg_actual_batch_size=avg_actual_bs,
        p50_actual_batch_size=p50_actual_bs,
        p95_actual_batch_size=p95_actual_bs,
        actual_batch_sizes=actual_batch_sizes,
        actual_batch_histogram=hist,
    )


async def run_continuous_batching(
    llm: LLM,
    requests: List[Request],
) -> BenchmarkResult:
    """
    continuous batching:
    - 请求按 arrival_time 到达
    - 到达就立刻 add_request
    - 引擎持续 step()
    - 新请求会和旧请求一起被 scheduler 调度
    """
    start_wall = time.perf_counter()
    done: List[Request] = []
    total_tokens = 0

    i = 0
    n = len(requests)
    active: Dict[int, Request] = {}

    while i < n or active or not llm.is_finished():
        now = time.perf_counter() - start_wall

        # 注入当前已到达请求
        while i < n and requests[i].arrival_time <= now:
            req = requests[i]
            sp = SamplingParams(
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            req.submit_time = now
            llm.add_request(req.prompt, sp)

            seq = llm.scheduler.waiting[-1]
            req.seq_id = seq.seq_id
            active[req.seq_id] = req
            i += 1

        # 如果当前引擎里有活跃请求，就 step
        if active or not llm.is_finished():
            outputs, _ = llm.step()
            now_finish = time.perf_counter() - start_wall

            for seq_id, token_ids in outputs:
                if seq_id in active:
                    req = active.pop(seq_id)
                    req.output_token_ids = token_ids
                    req.finish_time = now_finish
                    total_tokens += len(token_ids)
                    done.append(req)
        else:
            await asyncio.sleep(0.001)

        await asyncio.sleep(0)

    total_time = time.perf_counter() - start_wall
    latencies = [r.latency for r in done if r.latency is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return BenchmarkResult(
        strategy="continuous",
        num_requests=len(done),
        total_time=total_time,
        avg_latency=avg_latency,
        p50_latency=percentile(latencies, 50),
        p95_latency=percentile(latencies, 95),
        throughput_tokens=total_tokens / total_time if total_time > 0 else 0.0,
        throughput_requests=len(done) / total_time if total_time > 0 else 0.0,
    )


async def compare_strategies(
    model_path: str,
    prompts: List[str],
    arrival_rate: float = 5.0,
    jitter: float = 0.0,
    static_batch_sizes: List[int] = [1, 8, 16],
    flush_timeout: float = 0.2,
    max_tokens: int = 64,
    temperature: float = 0.7,
) -> List[BenchmarkResult]:
    base_requests = build_requests(
        prompts=prompts,
        arrival_rate=arrival_rate,
        jitter=jitter,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.info(f"Loading model: {model_path}")
    llm = LLM(model=model_path)

    results: List[BenchmarkResult] = []

    # warmup
    logger.info("Warmup...")
    warm_sp = SamplingParams(temperature=temperature, max_tokens=8)
    llm.generate(["warmup"], warm_sp, use_tqdm=False)

    # static
    for bs in static_batch_sizes:
        reqs = [
            Request(
                req_id=r.req_id,
                prompt=r.prompt,
                arrival_time=r.arrival_time,
                temperature=r.temperature,
                max_tokens=r.max_tokens,
            )
            for r in base_requests
        ]
        logger.info(f"Running static batching, batch_size={bs}")
        result = await run_static_batching(
            llm=llm,
            requests=reqs,
            batch_size=bs,
            flush_timeout=flush_timeout,
        )
        results.append(result)

    # continuous
    reqs = [
        Request(
            req_id=r.req_id,
            prompt=r.prompt,
            arrival_time=r.arrival_time,
            temperature=r.temperature,
            max_tokens=r.max_tokens,
        )
        for r in base_requests
    ]
    logger.info("Running continuous batching")
    result = await run_continuous_batching(llm=llm, requests=reqs)
    results.append(result)

    # llm.exit()
    return results


def print_results(results: List[BenchmarkResult]):
    print("\n" + "=" * 150)
    print("Batching Strategy Comparison")
    print("=" * 150)
    print(
        f"{'Strategy':<15} {'Total Time':<12} {'Avg Lat':<12} "
        f"{'P50 Lat':<12} {'P95 Lat':<12} {'Tok/s':<12} {'Req/s':<12} "
        f"{'Avg BS':<10} {'P50 BS':<10} {'P95 BS':<10}"
    )
    print("-" * 150)

    for r in results:
        print(
            f"{r.strategy:<15} "
            f"{r.total_time:<12.3f} "
            f"{r.avg_latency:<12.3f} "
            f"{r.p50_latency:<12.3f} "
            f"{r.p95_latency:<12.3f} "
            f"{r.throughput_tokens:<12.2f} "
            f"{r.throughput_requests:<12.2f} "
            f"{r.avg_actual_batch_size:<10.2f} "
            f"{r.p50_actual_batch_size:<10.2f} "
            f"{r.p95_actual_batch_size:<10.2f}"
        )

    print("=" * 150)

    # 额外打印 histogram
    print("\nActual batch size histogram:")
    for r in results:
        if r.actual_batch_histogram:
            print(f"{r.strategy}: {r.actual_batch_histogram}")


if __name__ == "__main__":
    import sys

    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing.",
        "Write a short poem about spring.",
        "What is 2 + 2?",
        "Describe photosynthesis.",
        "Benefits of exercise?",
        "Tell me a joke.",
    ] * 8

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6b-Instruct"

    results = asyncio.run(
        compare_strategies(
            model_path=model_path,
            prompts=test_prompts,
            arrival_rate=10.0,
            jitter=0.02,
            static_batch_sizes=[1, 8, 16,32],
            flush_timeout=0.5,
            max_tokens=256,
            temperature=0.7,
        )
    )
    print_results(results)