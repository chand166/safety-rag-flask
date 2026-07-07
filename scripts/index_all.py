"""
一键索引脚本 — 扫描知识库 → 提取文本 → 分块 → 写入向量数据库

用法:
  python scripts/index_all.py           # 正常索引
  python scripts/index_all.py --verbose  # 显示详细日志
"""
import sys
import os
from pathlib import Path

# 加入项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document_loader import load_all_documents
from src.text_splitter import split_documents
from src.vector_store import index_documents, get_stats


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("=" * 60)
    print("  安全培训知识库 — 文档索引")
    print("=" * 60)

    # 1. 加载文档
    print("\n📂 步骤 1: 扫描并提取文档...")
    docs = load_all_documents(verbose=verbose)

    total_chars = sum(d["char_count"] for d in docs)
    print(f"  成功加载 {len(docs)} 个文档, 总字符数: {total_chars:,}")

    # 2. 分块
    print("\n✂️  步骤 2: 文本分块...")
    chunks = split_documents(docs)
    print(f"  生成 {len(chunks)} 个文档片段")

    # 3. 索引
    print("\n💾 步骤 3: 写入向量数据库...")
    count = index_documents(chunks)

    # 4. 验证
    print("\n✅ 步骤 4: 验证索引...")
    stats = get_stats()
    print(f"  向量数据库状态:")
    print(f"  - 片段总数: {stats['total_chunks']}")
    print(f"  - 分类数: {stats['category_count']}")
    print(f"  - 分类: {', '.join(stats['categories'])}")

    print("\n" + "=" * 60)
    print("  ✓ 索引完成！")
    print(f"  启动服务: cd D:\\safety-rag && source venv/Scripts/activate && streamlit run src/main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()