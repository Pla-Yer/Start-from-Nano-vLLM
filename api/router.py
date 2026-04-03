"""
模型路由模块
根据请求内容选择合适的模型
"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ModelRouter:
    """模型路由器"""

    def __init__(self, config: dict):
        self.config = config
        routing_config = config.get("routing", {})
        self.default_model = routing_config.get("default_model", "qwen3-0.6b")

        # 按长度路由配置
        self.route_by_length = routing_config.get("route_by_length", {})
        self.short_threshold = self.route_by_length.get("short_threshold", 100)
        self.long_model = self.route_by_length.get("long_model", "qwen3-1.5b")

        # 路由统计
        self.route_stats = {}

    def route(self, prompt: str, model_hint: Optional[str] = None) -> str:
        """
        根据请求选择模型

        Args:
            prompt: 用户 prompt
            model_hint: 用户指定的模型（可选）

        Returns:
            模型名称
        """
        # 1. 如果用户指定了模型，直接使用
        if model_hint:
            self._record_route(model_hint)
            return model_hint

        # 2. 按 prompt 长度选择模型
        if self.route_by_length.get("enabled", False):
            prompt_len = len(prompt)
            if prompt_len < self.short_threshold:
                model = self.default_model
                logger.debug(f"Short prompt ({prompt_len} chars) -> {model}")
            else:
                model = self.long_model
                logger.debug(f"Long prompt ({prompt_len} chars) -> {model}")
            self._record_route(model)
            return model

        # 3. 默认模型
        self._record_route(self.default_model)
        return self.default_model

    def _record_route(self, model: str):
        """记录路由统计"""
        if model not in self.route_stats:
            self.route_stats[model] = 0
        self.route_stats[model] += 1

    def get_stats(self) -> dict:
        """获取路由统计"""
        return {
            "total_routes": sum(self.route_stats.values()),
            "by_model": self.route_stats,
            "default_model": self.default_model,
        }