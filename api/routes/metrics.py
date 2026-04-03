"""
指标路由
"""
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def get_metrics():
    """获取性能指标"""
    from api.main import metrics

    if not metrics:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return metrics.get_summary()


@router.get("/metrics/gpu")
async def get_gpu_metrics():
    """获取 GPU 指标"""
    from api.main import gpu_metrics

    if not gpu_metrics:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return gpu_metrics.get_gpu_stats()