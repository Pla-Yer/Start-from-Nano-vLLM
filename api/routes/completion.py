"""
Completion 路由
"""
import asyncio
import json
from datetime import datetime
from typing import List, AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.schemas import CompletionRequest, CompletionResponse

router = APIRouter(prefix="/v1", tags=["completion"])


@router.post("/completions")
async def completions(request: CompletionRequest):
    """Completion 接口 (OpenAI 兼容)"""
    from api.main import model_manager, model_router, metrics

    if not model_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # 选择模型
    model_name = request.model or model_router.default_model

    if model_name not in model_manager.loaded_models:
        await model_manager.load_model(model_name)

    if model_name not in model_manager.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    llm = model_manager.loaded_models[model_name]
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

    # 采样参数
    from nanovllm.sampling_params import SamplingParams
    sp = SamplingParams(
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 16,
    )

    start_time = asyncio.get_event_loop().time()

    try:
        if request.stream:
            return EventSourceResponse(
                generate_completion_stream(llm, prompts, sp, model_name),
                media_type="text/event-stream"
            )
        else:
            outputs = llm.generate(prompts, sp, use_tqdm=False)

            if not outputs:
                raise HTTPException(status_code=500, detail="Generation failed")

            choices = []
            total_prompt_tokens = 0
            total_completion_tokens = 0

            for i, output in enumerate(outputs):
                response_text = output["text"]
                completion_tokens = len(output["token_ids"])
                total_prompt_tokens += len(prompts[i])
                total_completion_tokens += completion_tokens

                choices.append({
                    "index": i,
                    "text": response_text,
                    "finish_reason": "stop"
                })

            latency = asyncio.get_event_loop().time() - start_time
            metrics.record_request(model_name, total_prompt_tokens, total_completion_tokens, latency)

            return CompletionResponse(
                id=f"cmpl-{datetime.now().timestamp()}",
                created=int(datetime.now().timestamp()),
                model=model_name,
                choices=choices,
                usage={
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens
                }
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_completion_stream(
    llm,
    prompts: List[str],
    sp,
    model_name: str
) -> AsyncGenerator[str, None]:
    """生成 Completion 流式响应"""
    outputs = llm.generate(prompts, sp, use_tqdm=False)

    if not outputs:
        return

    output = outputs[0]
    text = output["text"]

    response_id = f"cmpl-{datetime.now().timestamp()}"
    created = int(datetime.now().timestamp())

    for char in text:
        chunk = {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "text": char,
                "finish_reason": None
            }]
        }
        yield json.dumps(chunk)
        await asyncio.sleep(0.01)

    final_chunk = {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "text": "",
            "finish_reason": "stop"
        }]
    }
    yield json.dumps(final_chunk)
    yield "[DONE]"