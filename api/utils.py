"""
工具函数
"""
from typing import List
from api.schemas import Message


def messages_to_prompt(messages: List[Message]) -> str:
    """将 messages 转换为 prompt"""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    return "\n".join(parts)