"""
向量数据库模块 — 使用 ChromaDB 管理文档片段索引与检索
"""
import os
import time
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

from src.config_loader import config
from src.embeddings import compute_embeddings, embedding_dimension


# 自定义嵌入函数（适配 ChromaDB）
class LocalEmbeddingFunction:
    """ChromaDB 自定义嵌入函数，使用本地 sentence-transformers 模型"""

    def name(self):
        return "local_bge_zh"

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return compute_embeddings(input)

    def embed_query(self, input):
        """ChromaDB 查询嵌入"""
        if isinstance(input, str):
            return compute_embedding(input)
        return compute_embeddings(input)

    def embed_document(self, input):
        """ChromaDB 文档嵌入"""
        if isinstance(input, str):
            return compute_embedding(input)
        return compute_embeddings(input)


def get_client() -> chromadb.PersistentClient:
    """获取 ChromaDB 客户端"""
    persist_dir = config["vector_store"]["persist_directory"]
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client=None, collection_name: str = None):
    """获取或创建集合"""
    if client is None:
        client = get_client()
    if collection_name is None:
        collection_name = config["vector_store"]["collection_name"]

    emb_fn = LocalEmbeddingFunction()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )


def index_documents(chunks: List[Dict]) -> int:
    """
    将文档片段写入向量数据库。

    参数:
        chunks: 文档片段列表 [{"filepath", "relpath", "filename", "category",
                                "chunk_index", "chunk_total", "chunk_text", ...}, ...]

    返回:
        写入的片段数量
    """
    client = get_client()
    collection = get_collection(client)

    # 清空已有数据（重新索引）
    existing_count = collection.count()
    if existing_count > 0:
        print(f"  清空已有索引 ({existing_count} 条)...")
        all_ids = collection.get()["ids"]
        # 分批删除
        batch_size = 100
        for i in range(0, len(all_ids), batch_size):
            batch = all_ids[i:i + batch_size]
            collection.delete(ids=batch)

    # 批量写入
    batch_size = 50
    total = len(chunks)
    inserted = 0

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        ids = []
        metadatas = []
        documents = []

        for chunk in batch:
            # 唯一 ID
            uid = f"{chunk['relpath']}#{chunk['chunk_index']}"
            ids.append(uid)
            metadatas.append({
                "filepath": chunk["filepath"],
                "relpath": chunk["relpath"],
                "filename": chunk["filename"],
                "category": chunk["category"],
                "chunk_index": chunk["chunk_index"],
                "chunk_total": chunk["chunk_total"],
            })
            documents.append(chunk["chunk_text"])

        collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
        )
        inserted += len(batch)
        # 进度显示
        if (i // batch_size) % 5 == 0:
            print(f"  索引进度: {inserted}/{total}")

    print(f"  ✓ 索引完成: {inserted} 个片段")
    return inserted


def search(query: str, top_k: int = None) -> List[Dict]:
    """
    检索最相关的文档片段。

    返回:
    [
        {
            "id": "relative/path.pdf#3",
            "filepath": "D:\\...",
            "relpath": "relative/path.pdf",
            "filename": "文件.pdf",
            "category": "分类",
            "chunk_index": 3,
            "chunk_total": 10,
            "chunk_text": "片段内容...",
            "score": 0.85,
        },
        ...
    ]
    """
    if top_k is None:
        top_k = config["vector_store"]["retrieval"]["top_k"]

    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    hits = []
    if not results["ids"]:
        return hits

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        hits.append({
            "id": results["ids"][0][i],
            "filepath": meta.get("filepath", ""),
            "relpath": meta.get("relpath", ""),
            "filename": meta.get("filename", ""),
            "category": meta.get("category", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_total": meta.get("chunk_total", 0),
            "chunk_text": results["documents"][0][i],
            "score": results["distances"][0][i] if "distances" in results else 0,
        })

    return hits


def get_stats() -> Dict:
    """获取向量数据库统计信息"""
    collection = get_collection()
    count = collection.count()
    categories = set()
    for meta in collection.get(limit=count)["metadatas"]:
        categories.add(meta.get("category", "未分类"))
    return {
        "total_chunks": count,
        "categories": sorted(categories),
        "category_count": len(categories),
        "collection_name": config["vector_store"]["collection_name"],
    }


if __name__ == "__main__":
    from src.document_loader import load_all_documents
    from src.text_splitter import split_documents

    print("加载文档...")
    docs = load_all_documents(verbose=True)
    print(f"分割为片段...")
    chunks = split_documents(docs)
    print(f"得到 {len(chunks)} 个片段")
    print(f"索引到向量数据库...")
    index_documents(chunks)
    print(f"搜索测试...")
    results = search("实验室安全操作规程")
    for r in results[:3]:
        print(f"  [{r['score']:.3f}] {r['filename']} (第{r['chunk_index']+1}/{r['chunk_total']}段)")
        print(f"    {r['chunk_text'][:100]}...")
        print()