import os
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.file_download import model_file_download


def download_model_from_modelscope(
    model_id,
    local_dir,
    revision='master'
):
    """
    从 ModelScope 下载模型
    """
    try:
        print(f"正在从 ModelScope 下载模型 {model_id}...")
        
        # 展开用户路径
        local_dir = os.path.expanduser(local_dir)
        
        # 确保目标目录存在
        os.makedirs(local_dir, exist_ok=True)
        
        # 从 ModelScope 下载模型
        model_dir = snapshot_download(
            model_id=model_id,
            revision=revision,
            cache_dir=local_dir
        )
        
        print(f"模型成功下载到 {model_dir}")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False


if __name__ == "__main__":
    # 检查是否已安装 modelscope
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("未检测到 ModelScope，正在尝试安装...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
            from modelscope.hub.snapshot_download import snapshot_download
            print("ModelScope 安装成功")
        except Exception as e:
            print(f"ModelScope 安装失败: {e}")
            print("请手动安装: pip install modelscope")
            exit(1)
    
    success = download_model_from_modelscope(
        model_id="qwen/Qwen3-1.7B",  
        local_dir="~/huggingface/Qwen3-1.7B/", 
        revision='master'
    )
    
    if not success:
        print("下载失败，请检查网络连接或模型ID是否正确。")
        exit(1)