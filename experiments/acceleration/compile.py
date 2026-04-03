import json
import subprocess
import sys


def run_subprocess_case(script_path: str, model_path: str, mode: str, max_tokens: int = 64):
    cmd = [
        sys.executable,
        script_path,
        "--model", model_path,
        "--mode", mode,
        "--max-tokens", str(max_tokens),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Case failed: {mode}\n"
            f"returncode={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    # 取最后一个非空行作为 JSON 结果
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No output from case: {mode}")

    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON from case: {mode}\n"
            f"stdout:\n{proc.stdout}"
        ) from e

    return result


def print_results(results):
    eager = results["eager"]
    cudagraph = results["cudagraph"]

    print("\n" + "=" * 80)
    print("Eager vs CUDA Graph Benchmark")
    print("=" * 80)

    print("\n[eager / enforce_eager=True]")
    print(f"  Total time: {eager['total_time']:.3f}s")
    print(f"  Throughput (tok/s): {eager['throughput_tokens']:.2f}")
    print(f"  Throughput (req/s): {eager['throughput_requests']:.2f}")

    print("\n[cudagraph / enforce_eager=False]")
    print(f"  Total time: {cudagraph['total_time']:.3f}s")
    print(f"  Throughput (tok/s): {cudagraph['throughput_tokens']:.2f}")
    print(f"  Throughput (req/s): {cudagraph['throughput_requests']:.2f}")

    if eager["total_time"] > 0 and eager["throughput_tokens"] > 0:
        time_reduction = (eager["total_time"] - cudagraph["total_time"]) / eager["total_time"] * 100
        tok_increase = (
            (cudagraph["throughput_tokens"] - eager["throughput_tokens"])
            / eager["throughput_tokens"] * 100
        )

        print("\n[improvement: cudagraph vs eager]")
        print(f"  Time reduction: {time_reduction:.1f}%")
        print(f"  Throughput increase: {tok_increase:.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6b-Instruct"
    case_script = sys.argv[2] if len(sys.argv) > 2 else "cudagraph.py"

    results = {}
    results["cudagraph"] = run_subprocess_case(case_script, model_path, "cudagraph")
    results["eager"] = run_subprocess_case(case_script, model_path, "eager")

    print_results(results)