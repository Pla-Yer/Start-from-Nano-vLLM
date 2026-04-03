"""
模型管理模块
支持多模型加载、热卸载、动态切换
"""
import asyncio
import threading
from typing import Dict, Optional
import logging
import torch
import gc

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
        except AssertionError as e:
            # AssertionError 通常表示 KV cache 分配失败（内存不足）
            self.loading_models.pop(model_name, None)
            logger.error(f"❌ AssertionError while loading model {model_name}: {e}")
            logger.error(f"   This usually means not enough GPU memory for KV cache allocation")

            # 清理 GPU 缓存
            self._cleanup_gpu_memory()

            # 转换为 OOM 错误，让上层统一处理
            raise torch.cuda.OutOfMemoryError(
                f"Not enough GPU memory to allocate KV cache for model {model_name}. "
                f"Please try unloading other models or using a smaller model."
            )
        except torch.cuda.OutOfMemoryError as e:
            # OOM 错误：清理并重新抛出
            self.loading_models.pop(model_name, None)
            logger.error(f"❌ OOM while loading model {model_name}")
            logger.error(f"   Error: {str(e)[:200]}")  # 只显示前200字符

            # 清理 GPU 缓存
            self._cleanup_gpu_memory()

            # 重新抛出异常，让上层处理
            raise
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

            # 调用 exit() 方法正确清理资源
            cleanup_error = None
            try:
                logger.info(f"🧹 Cleaning up model {model_name}...")

                # 手动清理模型组件（使用 getattr 避免属性不存在错误）
                if hasattr(llm, 'model_runner') and llm.model_runner is not None:
                    runner = llm.model_runner

                    # 清理模型权重（最大的内存占用）
                    model = getattr(runner, 'model', None)
                    if model is not None:
                        # 将模型移到 CPU 再删除，避免 CUDA 内存碎片
                        try:
                            logger.info(f"   Moving model weights to CPU...")
                            for param in model.parameters():
                                param.data = param.data.cpu()
                        except Exception as e:
                            logger.warning(f"   Failed to move model to CPU: {e}")
                        finally:
                            try:
                                del runner.model
                            except:
                                pass

                    # 清理 KV cache
                    kv_cache = getattr(runner, 'kv_cache', None)
                    if kv_cache is not None:
                        logger.info(f"   Deleting KV cache...")
                        try:
                            del runner.kv_cache
                        except Exception as e:
                            logger.warning(f"   Failed to delete KV cache: {e}")

                    # 清理 CUDA graphs（只在非 eager 模式下存在）
                    graphs = getattr(runner, 'graphs', None)
                    if graphs is not None:
                        logger.info(f"   Deleting CUDA graphs...")
                        try:
                            del runner.graphs
                        except Exception as e:
                            logger.warning(f"   Failed to delete graphs: {e}")

                    graph_pool = getattr(runner, 'graph_pool', None)
                    if graph_pool is not None:
                        try:
                            del runner.graph_pool
                        except:
                            pass

                    # 清理 sampler
                    sampler = getattr(runner, 'sampler', None)
                    if sampler is not None:
                        try:
                            del runner.sampler
                        except:
                            pass

                    # 清理 graph_vars
                    graph_vars = getattr(runner, 'graph_vars', None)
                    if graph_vars is not None:
                        try:
                            del runner.graph_vars
                        except:
                            pass

                # 调用官方 exit 方法（会清理分布式资源）
                logger.info(f"🧹 Calling exit() for model {model_name}...")
                llm.exit()

            except Exception as e:
                cleanup_error = e
                logger.error(f"❌ Critical error during model cleanup: {e}")
                import traceback
                logger.error(traceback.format_exc())

            # 删除模型对象
            try:
                del llm
            except:
                pass

            # 清理 GPU 内存
            self._cleanup_gpu_memory()

            logger.info(f"🗑️ Model {model_name} unloaded successfully")
            return True

    async def unload_all(self):
        """卸载所有模型"""
        # 先获取所有模型名称（在锁内）
        with self.lock:
            model_names = list(self.loaded_models.keys())

        # 逐个卸载（在锁外，避免死锁）
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

    def _cleanup_gpu_memory(self):
        """清理 GPU 内存"""
        logger.info("🧹 Cleaning up GPU memory...")
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            # 打印内存使用情况
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"   GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

    async def handle_oom_and_retry(self, model_name: str, model_config: dict) -> bool:
        """
        处理 OOM 并尝试重新加载模型
        策略：卸载其他模型，然后重试加载
        """
        logger.warning(f"🔥 OOM detected, attempting to free memory for {model_name}...")

        # 打印当前内存状态
        self._cleanup_gpu_memory()

        # 获取当前加载的其他模型
        other_models = [name for name in self.loaded_models.keys() if name != model_name]

        if not other_models:
            logger.error(f"❌ No other models to unload, cannot load {model_name}")
            raise RuntimeError(f"Not enough GPU memory to load model {model_name}")

        logger.info(f"📋 Found {len(other_models)} other model(s) to unload: {other_models}")

        # 卸载其他模型（从最旧的开始）
        for i, other_model in enumerate(other_models):
            logger.warning(f"🗑️ [{i+1}/{len(other_models)}] Unloading model {other_model} to free memory...")
            await self.unload_model(other_model)

            # 打印卸载后的内存状态
            logger.info(f"📊 Memory status after unloading {other_model}:")
            self._cleanup_gpu_memory()

            # 尝试重新加载目标模型
            try:
                logger.info(f"🔄 Retrying to load model {model_name}...")
                await self.load_model(model_name)
                logger.info(f"✅ Successfully loaded {model_name} after unloading {other_model}")
                return True
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"⚠️ Still OOM after unloading {other_model}, trying next model...")
                continue
            except Exception as e:
                logger.error(f"❌ Unexpected error while loading {model_name}: {e}")
                raise

        # 所有模型都卸载了还是 OOM
        logger.error(f"❌ Cannot load {model_name} even after unloading all other models")
        raise RuntimeError(f"Not enough GPU memory to load model {model_name}")