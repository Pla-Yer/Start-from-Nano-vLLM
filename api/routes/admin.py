"""
管理路由
"""
import os
import yaml

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/reload-config")
async def reload_config():
    """重新加载配置"""
    from api.main import model_manager, model_router, ModelRouter

    config_path = os.path.join(os.path.dirname(__file__), "../../config/models.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_manager.config = config
    model_router = ModelRouter(config)

    return {"status": "reloaded"}


@router.post("/unload/{model_name:path}")
async def unload_model(model_name: str):
    """卸载模型"""
    from api.main import model_manager

    if not model_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await model_manager.unload_model(model_name)
    return {"status": "success" if result else "not_loaded", "model": model_name}