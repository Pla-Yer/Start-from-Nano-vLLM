"""
API 测试脚本
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_root():
    """测试根路径"""
    print("\n=== 测试根路径 ===")
    r = requests.get(f"{BASE_URL}/")
    print(f"状态码: {r.status_code}")
    print(f"内容类型: {r.headers.get('content-type')}")
    if "text/html" in r.headers.get("content-type", ""):
        print("✅ 返回 HTML 页面")
        return True
    return False


def test_health():
    """测试健康检查"""
    print("\n=== 测试健康检查 ===")
    r = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {r.status_code}")
    data = r.json()
    print(f"响应: {data}")
    if data.get("status") == "healthy":
        print("✅ 服务健康")
        return True
    return False


def test_list_models():
    """测试列出模型"""
    print("\n=== 测试列出模型 ===")
    r = requests.get(f"{BASE_URL}/v1/models")
    print(f"状态码: {r.status_code}")
    data = r.json()
    print(f"模型列表: {json.dumps(data, indent=2, ensure_ascii=False)}")
    if "data" in data:
        print(f"✅ 可用模型: {[m['id'] for m in data['data']]}")
        return True
    return False


def test_chat_completion():
    """测试 Chat Completion"""
    print("\n=== 测试 Chat Completion ===")

    payload = {
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "user", "content": "你好"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"状态码: {r.status_code}")

    if r.status_code != 200:
        print(f"❌ 错误: {r.text}")
        return False

    data = r.json()
    print(f"ID: {data.get('id')}")
    print(f"模型: {data.get('model')}")

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    print(f"回复: {message.get('content')}")

    usage = data.get("usage", {})
    print(f"Token 使用: {usage}")

    if message.get("content"):
        print("✅ Chat Completion 正常")
        return True
    return False


def test_chat_stream():
    """测试流式 Chat Completion"""
    print("\n=== 测试流式 Chat Completion ===")

    payload = {
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "user", "content": "你好"}
        ],
        "stream": True,
        "max_tokens": 30
    }

    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True
    )

    print(f"状态码: {r.status_code}")

    if r.status_code != 200:
        print(f"❌ 错误: {r.text}")
        return False

    print("流式响应:")
    full_content = ""
    for line in r.iter_lines():
        line = line.decode("utf-8").strip()
        if not line:
            continue
        if line == "data: [DONE]":
            break
        if line.startswith("data: "):
            try:
                chunk = json.loads(line[6:])
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    full_content += content
                    print(content, end="", flush=True)
            except:
                pass

    print("\n")
    if full_content:
        print(f"✅ 流式响应正常 (收到 {len(full_content)} 字符)")
        return True
    return False


def test_completion():
    """测试 Completion"""
    print("\n=== 测试 Completion ===")

    payload = {
        "model": "qwen3-0.6b",
        "prompt": "你好，请介绍一下自己",
        "temperature": 0.7,
        "max_tokens": 100
    }

    r = requests.post(
        f"{BASE_URL}/v1/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"状态码: {r.status_code}")

    if r.status_code != 200:
        print(f"❌ 错误: {r.text}")
        return False

    data = r.json()
    choice = data.get("choices", [{}])[0]
    print(f"回复: {choice.get('text')}")

    if choice.get("text"):
        print("✅ Completion 正常")
        return True
    return False


def test_metrics():
    """测试指标"""
    print("\n=== 测试指标 ===")
    r = requests.get(f"{BASE_URL}/metrics")
    print(f"状态码: {r.status_code}")
    data = r.json()
    print(f"指标: {json.dumps(data, indent=2)}")
    if "total_requests" in data:
        print("✅ 指标正常")
        return True
    return False


def main():
    print(f"测试 nano-vllm API ({BASE_URL})")
    print("=" * 50)

    results = []

    results.append(("根路径", test_root()))
    results.append(("健康检查", test_health()))
    results.append(("模型列表", test_list_models()))
    results.append(("Chat Completion", test_chat_completion()))
    results.append(("流式 Chat", test_chat_stream()))
    results.append(("Completion", test_completion()))
    results.append(("指标", test_metrics()))

    print("\n" + "=" * 50)
    print("测试结果:")
    passed = 0
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
        if ok:
            passed += 1

    print(f"\n通过: {passed}/{len(results)}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())