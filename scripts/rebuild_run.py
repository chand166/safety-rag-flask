import sys
sys.path.insert(0, ".")
from src.document_loader import load_all_documents
from src.text_splitter import split_documents
from src.vector_store import index_documents, get_stats
print("加载文档...")
docs = load_all_documents(verbose=True)
print(f"文档数: {len(docs)}")
print("分块...")
chunks = split_documents(docs)
print(f"片段数: {len(chunks)}")
print("索引...")
count = index_documents(chunks)
print(f"索引完成: {count} 片段")
stats = get_stats()
print(f"总片段: {stats}")