"""
LLM 客户端模块 — 支持多个模型内网 API

配置结构（config.yaml）:
  llm:
    default_model: "kimi-k2.5"
    models:
      kimi-k2.5:
        name: "Kimi-K2.5"
        api_base: "http://10.2.39.3:1025/v1"
        api_key: "EMPTY"
        model: "Kimi-K2.5"
        temperature: 0.3
        max_tokens: 1024
      minimax-25:
        name: "MiniMax-25"
        api_base: "http://10.2.39.6:20004/v1"
        api_key: "EMPTY"
        model: "minimax25"
        temperature: 0.8
        max_tokens: 512
"""
from typing import List, Dict, Optional

from openai import OpenAI
from src.config_loader import config


class LLMClient:
    """多模型 LLM API 客户端"""

    def __init__(self, model_key: str = None):
        llm_cfg = config["llm"]
        # 确定模型 key
        if model_key is None:
            model_key = llm_cfg.get("default_model", "kimi-k2.5")

        # 查找模型配置
        models = llm_cfg.get("models", {})
        if model_key not in models:
            raise ValueError(f"未知模型: {model_key}，可用: {list(models.keys())}")

        m = models[model_key]
        self.model_key = model_key
        self.model_id = m["model"]
        self.client = OpenAI(
            api_key=m["api_key"],
            base_url=m["api_base"],
            timeout=180.0,
            max_retries=1,
        )
        self.temperature = m.get("temperature", 0.3)
        self.max_tokens = m.get("max_tokens", 2048)

    def chat(self, messages: List[Dict], stream: bool = False) -> str:
        """
        发送聊天请求。

        参数:
            messages: OpenAI 格式消息列表
            stream: 是否流式输出

        返回:
            模型回复文本
        """
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream,
        )
        if stream:
            full_text = ""
            for chunk in resp:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_text += delta.content
            return full_text
        else:
            return resp.choices[0].message.content


# 模型缓存
_llm_cache: Dict[str, LLMClient] = {}


def get_llm(model_key: str = None) -> LLMClient:
    """获取 LLM 客户端。默认返回 default_model，指定则切换模型。"""
    if model_key is None:
        model_key = config["llm"].get("default_model", "kimi-k2.5")

    if model_key not in _llm_cache:
        _llm_cache[model_key] = LLMClient(model_key)
    return _llm_cache[model_key]


def get_available_models() -> List[Dict]:
    """返回可用模型列表（给前端用）"""
    models = config["llm"].get("models", {})
    default = config["llm"].get("default_model", "kimi-k2.5")
    result = []
    for key, m in models.items():
        result.append({
            "key": key,
            "name": m.get("name", key),
            "is_default": key == default,
        })
    return result


if __name__ == "__main__":
    # 测试
    models = get_available_models()
    print("可用模型:")
    for m in models:
        print(f"  {m['key']}: {m['name']}{' (默认)' if m['is_default'] else ''}")

    print("\n--- 测试 Kimi-K2.5 ---")
    client = get_llm("kimi-k2.5")
    resp = client.chat([
        {"role": "system", "content": "你是一个有帮助的助手。一句话回答。"},
        {"role": "user", "content": "你好，请介绍一下你自己。"},
    ])
    print(f"回复: {resp}")

    print("\n--- 测试 MiniMax-25 ---")
    client2 = get_llm("minimax-25")
    resp2 = client2.chat([
        {"role": "system", "content": "你是一个有帮助的助手。一句话回答。"},
        {"role": "user", "content": "你好，请介绍一下你自己。"},
    ])
    print(f"回复: {resp2}")