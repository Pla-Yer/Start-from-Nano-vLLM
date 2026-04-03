"""
健康检查路由
"""
from datetime import datetime
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """健康检查"""
    from api.main import model_manager

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(model_manager.loaded_models.keys()) if model_manager else []
    }