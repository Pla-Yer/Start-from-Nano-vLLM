"""
模型路由
"""
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models():
    """列出可用模型 (OpenAI 兼容)"""
    from api.main import model_manager

    if not model_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    models = []
    for name, config in model_manager.config["models"].items():
        is_loaded = name in model_manager.loaded_models
        models.append({
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "nano-vllm",
            "permission": [],
        })

    return {"object": "list", "data": models}


@router.get("/models/{model_name:path}")
async def get_model(model_name: str):
    """获取特定模型信息"""
    from api.main import model_manager

    if not model_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if model_name not in model_manager.config["models"]:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    is_loaded = model_name in model_manager.loaded_models

    return {
        "id": model_name,
        "object": "model",
        "created": 0,
        "owned_by": "nano-vllm",
        "permission": [],
        "loaded": is_loaded,
    }