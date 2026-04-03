"""
模型管理模块
支持多模型加载、热卸载、动态切换
"""
import asyncio
import threading
from typing import Dict, Optional
import logging

from nanovllm import LLM

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器"""

    def __init__(self, config: dict):
        self.config = config
        self.loaded_models: Dict[str, LLM] = {}
        self.loading_models: Dict[str, asyncio.Event] = {}
        self.lock = threading.Lock()

    async def load_model(self, model_name: str) -> bool:
        """加载模型"""
        with self.lock:
            # 已加载
            if model_name in self.loaded_models:
                return True

            # 正在加载
            if model_name in self.loading_models:
                event = self.loading_models[model_name]
            else:
                # 检查配置
                if model_name not in self.config["models"]:
                    raise ValueError(f"Model {model_name} not in config")

                model_config = self.config["models"][model_name]
                if not model_config.get("enabled", True):
                    raise ValueError(f"Model {model_name} is disabled")

                # 开始加载
                event = asyncio.Event()
                self.loading_models[model_name] = event

                # 在线程中加载模型（nano-vllm 初始化是同步的）
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model_sync, model_name, model_config)

                # 标记加载完成
                self.loaded_models[model_name] = self.loading_models.pop(model_name)
                event.set()

                logger.info(f"✅ Model {model_name} loaded successfully")
                return True

        # 等待其他任务加载完成
        await event.wait()
        return True

    def _load_model_sync(self, model_name: str, model_config: dict):
        """同步加载模型（在线程中执行）"""
        try:
            llm = LLM(
                model=model_config["path"],
                max_num_batched_tokens=model_config.get("max_num_batched_tokens", 16384),
                max_num_seqs=model_config.get("max_num_seqs", 256),
                max_model_len=model_config.get("max_model_len", 4096),
                gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.9),
                tensor_parallel_size=model_config.get("tensor_parallel_size", 1),
                enforce_eager=model_config.get("enforce_eager", False),
            )
            self.loading_models[model_name] = llm
        except Exception as e:
            self.loading_models.pop(model_name, None)
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise

    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        with self.lock:
            if model_name not in self.loaded_models:
                return False

            llm = self.loaded_models.pop(model_name)
            # nano-vllm 的 LLM 在退出时会自动清理
            # 这里可以添加额外的清理逻辑

            logger.info(f"🗑️ Model {model_name} unloaded")
            return True

    async def unload_all(self):
        """卸载所有模型"""
        with self.lock:
            model_names = list(self.loaded_models.keys())
            for name in model_names:
                await self.unload_model(name)

    def get_model(self, model_name: str) -> Optional[LLM]:
        """获取已加载的模型"""
        return self.loaded_models.get(model_name)

    def list_loaded(self) -> list:
        """列出已加载的模型"""
        return list(self.loaded_models.keys())

    def is_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        return model_name in self.loaded_models