"""
Pydantic 模型定义
"""
from typing import List, Optional, Union
from pydantic import BaseModel, Field


# ========== Request Models ==========

class Message(BaseModel):
    """Chat message"""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion 请求"""
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    """OpenAI Completion 请求"""
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 16
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


# ========== Response Models ==========

class UsageInfo(BaseModel):
    """Token 使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatMessage(BaseModel):
    """Chat 消息"""
    role: str
    content: str


class ChatChoice(BaseModel):
    """Chat 选择"""
    index: int
    message: ChatMessage
    finish_reason: str


class CompletionChoice(BaseModel):
    """Completion 选择"""
    text: str
    index: int
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """Chat Completion 响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo


class CompletionResponse(BaseModel):
    """Completion 响应"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo