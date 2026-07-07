"""
嵌入模块 — 使用本地 sentence-transformers 模型生成文本向量
"""
import os
from typing import List

# 在导入 sentence_transformers 之前设置镜像
_mirror = None
try:
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    _mirror = _cfg.get("embedding", {}).get("hf_mirror")
except Exception:
    pass

if _mirror:
    os.environ.setdefault("HF_ENDPOINT", _mirror)

from sentence_transformers import SentenceTransformer
from src.config_loader import config


# 单例模型
_MODEL = None


def get_embedding_model():
    """获取或初始化嵌入模型（单例模式）"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_cfg = config["embedding"]
    model_name = model_cfg["model_name"]
    device = model_cfg.get("device", "cpu")

    print(f"加载嵌入模型: {model_name} (device={device})")
    _MODEL = SentenceTransformer(model_name, device=device)
    print(f"  ✓ 模型维度: {_MODEL.get_sentence_embedding_dimension()}")
    return _MODEL


def compute_embedding(text: str) -> List[float]:
    """计算单条文本的嵌入向量"""
    model = get_embedding_model()
    emb = model.encode(f"为这个句子生成表示：{text}", normalize_embeddings=True)
    return emb.tolist()


def compute_embeddings(texts: List[str]) -> List[List[float]]:
    """批量计算文本嵌入向量"""
    model = get_embedding_model()
    texts_with_prefix = [f"为这个句子生成表示：{t}" for t in texts]
    embs = model.encode(texts_with_prefix, normalize_embeddings=True, show_progress_bar=True)
    return [e.tolist() for e in embs]


def embedding_dimension() -> int:
    """返回嵌入向量维度"""
    model = get_embedding_model()
    return model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    texts = ["实验室安全守则", "化学品管理规定", "应急处理方案"]
    embs = compute_embeddings(texts)
    print(f"维度: {len(embs[0])}")
    for t, e in zip(texts, embs):
        print(f"  '{t}' -> {e[:5]}...")