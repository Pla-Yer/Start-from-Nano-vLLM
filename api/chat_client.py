#!/usr/bin/env python3
"""
使用 OpenAI 库风格的对话客户端
"""
import os
# os.environ.pop("http_proxy", None)
# os.environ.pop("https_proxy", None)
# os.environ.pop("HTTP_PROXY", None)
# os.environ.pop("HTTPS_PROXY", None)
# os.environ.pop("all_proxy", None)
# os.environ.pop("ALL_PROXY", None)

from openai import OpenAI

# 配置 API
client = OpenAI(
    api_key="sk-dummy",
    base_url="http://localhost:8000/v1"
)

# 默认模型
MODEL = "qwen3-0.6b"


def chat(prompt: str, system: str = None, stream: bool = True):
    """单轮对话"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if stream:
        print("助手: ", end="")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            max_tokens=512,
            temperature=0.7,
        )

        content = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                content += delta.content
        print()
        return content
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        print(f"助手: {content}")
        return content


def chat_loop():
    """交互式对话循环"""
    print("=" * 50)
    print("  nano-vllm 对话客户端 (OpenAI 风格)")
    print("  输入 /exit 退出, /clear 清屏")
    print("=" * 50)
    print()

    messages = []

    while True:
        try:
            user_input = input("你: ").strip()

            if not user_input:
                continue

            if user_input == "/exit":
                print("再见!")
                break

            if user_input == "/clear":
                messages = []
                print("对话已清空")
                continue

            # 添加用户消息
            messages.append({"role": "user", "content": user_input})

            # 流式输出
            print("助手: ", end="", flush=True)
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
                max_tokens=512,
                temperature=0.7,
            )

            content = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    content += delta.content
            print()

            # 添加助手消息
            messages.append({"role": "assistant", "content": content})

            # 限制历史长度
            if len(messages) > 20:
                messages = messages[-20:]

        except KeyboardInterrupt:
            print("\n使用 /exit 退出")
        except EOFError:
            break


if __name__ == "__main__":
    # 简单测试
    print("测试单轮对话:")
    chat("用一句话介绍自己")

    print("\n" + "=" * 50)
    print("进入交互式对话 (按 Ctrl+C 退出)")
    print("=" * 50)

    try:
        chat_loop()
    except KeyboardInterrupt:
        print("\n再见!")