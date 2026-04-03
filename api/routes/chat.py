"""
Chat Completion 路由
"""
import asyncio
import json
from datetime import datetime
from typing import List, AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.schemas import ChatCompletionRequest, ChatCompletionResponse
from api.utils import messages_to_prompt

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat Completion 接口 (OpenAI 兼容)"""
    from api.main import model_manager, model_router, metrics

    if not model_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # 选择模型
    model_name = request.model or model_router.route(
        messages_to_prompt(request.messages),
        request.model
    )

    if model_name not in model_manager.loaded_models:
        await model_manager.load_model(model_name)

    if model_name not in model_manager.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    llm = model_manager.loaded_models[model_name]
    prompt = messages_to_prompt(request.messages)

    # 采样参数
    from nanovllm.sampling_params import SamplingParams
    sp = SamplingParams(
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 1024,
    )

    start_time = asyncio.get_event_loop().time()

    try:
        if request.stream:
            return EventSourceResponse(
                generate_chat_stream(llm, prompt, sp, model_name),
                media_type="text/event-stream"
            )
        else:
            outputs = llm.generate([prompt], sp, use_tqdm=False)

            if not outputs:
                raise HTTPException(status_code=500, detail="Generation failed")

            output = outputs[0]
            response_text = output["text"]
            completion_tokens = len(output["token_ids"])

            latency = asyncio.get_event_loop().time() - start_time
            metrics.record_request(model_name, len(prompt), completion_tokens, latency)

            return ChatCompletionResponse(
                id=f"chatcmpl-{datetime.now().timestamp()}",
                created=int(datetime.now().timestamp()),
                model=model_name,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": len(prompt),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(prompt) + completion_tokens
                }
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_chat_stream(
    llm,
    prompt: str,
    sp,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """生成流式响应"""
    from nanovllm.sampling_params import SamplingParams

    sp_stream = SamplingParams(
        temperature=sp.temperature,
        max_tokens=sp.max_tokens,
    )

    outputs = llm.generate([prompt], sp_stream, use_tqdm=False)

    if not outputs:
        return

    output = outputs[0]
    text = output["text"]

    response_id = f"chatcmpl-{datetime.now().timestamp()}"
    created = int(datetime.now().timestamp())

    for char in text:
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": char},
                "finish_reason": None
            }]
        }
        yield json.dumps(chunk)
        await asyncio.sleep(0.01)

    # 发送最后的 chunk
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield json.dumps(final_chunk)
    yield "[DONE]"