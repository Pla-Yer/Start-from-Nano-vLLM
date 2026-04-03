"""
性能监控模块
收集和展示推理性能指标
"""
import time
import threading
from collections import defaultdict, deque
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """性能指标收集器"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

        # 请求级别指标
        self.requests: deque = deque(maxlen=window_size)

        # 聚合统计
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency = 0.0

        # 按模型统计
        self.model_stats: Dict[str, dict] = defaultdict(lambda: {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latencies": deque(maxlen=window_size),
        })

        # 锁
        self._lock = threading.Lock()

    def record_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
        latency: float = 0.0,
    ):
        """记录单个请求"""
        with self._lock:
            self.total_requests += 1
            self.total_tokens += prompt_tokens + completion_tokens
            self.total_latency += latency

            # 按模型统计
            stats = self.model_stats[model]
            stats["requests"] += 1
            stats["prompt_tokens"] += prompt_tokens
            stats["completion_tokens"] += completion_tokens
            if latency > 0:
                stats["latencies"].append(latency)

            # 记录请求详情
            self.requests.append({
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency": latency,
                "timestamp": time.time(),
            })

    def record_latency(self, model: str, latency: float):
        """记录延迟"""
        with self._lock:
            if model in self.model_stats:
                self.model_stats[model]["latencies"].append(latency)

    def get_summary(self) -> dict:
        """获取指标摘要"""
        with self._lock:
            avg_latency = (
                self.total_latency / self.total_requests
                if self.total_requests > 0
                else 0
            )

            # 计算吞吐量
            throughput = (
                self.total_tokens / self.total_latency
                if self.total_latency > 0
                else 0
            )

            # 按模型汇总
            models = {}
            for model, stats in self.model_stats.items():
                latencies = list(stats["latencies"])
                models[model] = {
                    "requests": stats["requests"],
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                    "p50_latency": self._percentile(latencies, 0.5),
                    "p95_latency": self._percentile(latencies, 0.95),
                    "p99_latency": self._percentile(latencies, 0.99),
                }

            return {
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "avg_latency": avg_latency,
                "throughput_tokens_per_sec": throughput,
                "by_model": models,
            }

    def _percentile(self, values: list, p: float) -> float:
        """计算百分位数"""
        if not values:
            return 0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def reset(self):
        """重置指标"""
        with self._lock:
            self.requests.clear()
            self.total_requests = 0
            self.total_tokens = 0
            self.total_latency = 0.0
            self.model_stats.clear()


class GPUMetricsCollector:
    """GPU 指标收集器"""

    def __init__(self):
        self.has_gpu = False
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            self.torch = torch
        except ImportError:
            pass

    def get_gpu_stats(self) -> dict:
        """获取 GPU 统计"""
        if not self.has_gpu:
            return {"available": False}

        stats = {
            "available": True,
            "devices": [],
        }

        for i in range(self.torch.cuda.device_count()):
            mem_allocated = self.torch.cuda.memory_allocated(i) / 1024**3  # GB
            mem_reserved = self.torch.cuda.memory_reserved(i) / 1024**3  # GB
            mem_total = self.torch.cuda.get_device_properties(i).total_memory / 1024**3

            stats["devices"].append({
                "id": i,
                "name": self.torch.cuda.get_device_name(i),
                "memory_allocated_gb": round(mem_allocated, 2),
                "memory_reserved_gb": round(mem_reserved, 2),
                "memory_total_gb": round(mem_total, 2),
                "utilization_percent": round(mem_allocated / mem_total * 100, 1),
            })

        return stats


# 全局收集器
_global_metrics: Optional[MetricsCollector] = None
_gpu_metrics: Optional[GPUMetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def get_gpu_metrics() -> GPUMetricsCollector:
    """获取全局 GPU 指标收集器"""
    global _gpu_metrics
    if _gpu_metrics is None:
        _gpu_metrics = GPUMetricsCollector()
    return _gpu_metrics