"""
文本分块模块 — 将长文档按大小切割为有重叠的段落
"""
import re
from typing import List, Dict

from src.config_loader import config


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    将长文本切割为带重叠的段落块。

    策略:
    1. 优先按双换行（段落边界）分割
    2. 长段落内部按句号、问号、感叹号分割
    3. 超长无标点文本按字符硬切
    """
    if chunk_size is None:
        chunk_size = config["documents"]["chunk"]["size"]
    if chunk_overlap is None:
        chunk_overlap = config["documents"]["chunk"]["overlap"]

    text = text.strip()
    if not text:
        return []

    # 第一步：按段落（双换行或换行）分块
    # 保留常见的分段标记
    paragraphs = re.split(r"\n\s*\n+", text)
    # 也处理单换行（很多 PDF 提取出来是单换行）
    # 但保留明确的分段

    chunks = []
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果段落很短，积累到 buffer
        if len(buffer) + len(para) < chunk_size * 1.5:
            if buffer:
                buffer += "\n" + para
            else:
                buffer = para
            continue

        # buffer 够长了，或者当前段落很长
        if buffer:
            chunks.append(buffer)
            buffer = ""

        # 超长段落内部再分割
        if len(para) > chunk_size:
            sub_chunks = _split_long_paragraph(para, chunk_size)
            chunks.extend(sub_chunks)
        else:
            chunks.append(para)

    if buffer:
        chunks.append(buffer)

    # 第二步：应用重叠
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, chunk_overlap)

    return chunks


def _split_long_paragraph(text: str, chunk_size: int) -> List[str]:
    """将超长段落按句子边界分割"""
    # 按句号、问号、感叹号、分号、换行分割
    sentences = re.split(r"(?<=[。！？；\n])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    buffer = ""
    for sent in sentences:
        if len(buffer) + len(sent) > chunk_size and buffer:
            chunks.append(buffer)
            buffer = sent
        else:
            if buffer:
                buffer += sent
            else:
                buffer = sent

    if buffer:
        # 如果最后一段仍然超长，强制截断
        while len(buffer) > chunk_size:
            chunks.append(buffer[:chunk_size])
            buffer = buffer[chunk_size:]
        if buffer:
            chunks.append(buffer)

    return chunks


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """在相邻块之间添加重叠文本"""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        curr = chunks[i]
        # 从上一块末尾取 overlap 字符作为前缀
        if len(prev) > overlap:
            prefix = prev[-overlap:]
            # 尽量在标点处截断
            punct_match = re.search(r"[。！？；，、]", prefix)
            if punct_match:
                prefix = prefix[punct_match.end():]
            result.append(prefix + curr)
        else:
            result.append(curr)
    return result


def split_documents(loaded_docs: List[Dict]) -> List[Dict]:
    """
    将加载的文档列表全部切分为块。

    输入: [{"filepath": ..., "relpath": ..., "text": ..., ...}, ...]
    输出: [{"filepath": ..., "relpath": ..., "filename": ..., "category": ...,
             "chunk_index": 0, "chunk_text": "...", "chunk_total": 5}, ...]
    """
    all_chunks = []
    for doc in loaded_docs:
        text = doc.get("text", "")
        if not text:
            continue
        chunks = chunk_text(text)
        for idx, chunk_text_content in enumerate(chunks):
            all_chunks.append({
                "filepath": doc["filepath"],
                "relpath": doc["relpath"],
                "filename": doc["filename"],
                "category": doc["category"],
                "chunk_index": idx,
                "chunk_total": len(chunks),
                "chunk_text": chunk_text_content,
                "char_count": len(chunk_text_content),
            })
    return all_chunks


if __name__ == "__main__":
    sample = (
        "第一章 总则\n\n"
        "第一条 为了加强实验室安全管理，保障人身和财产安全，维护教学、科研工作的正常秩序，"
        "根据《中华人民共和国安全生产法》等法律法规，制定本规范。\n\n"
        "第二条 本规范适用于高等学校实验室的安全管理。\n\n"
        "第三条 实验室安全管理应坚持安全第一、预防为主、综合治理的方针。"
    )
    chunks = chunk_text(sample, chunk_size=100, chunk_overlap=20)
    for i, c in enumerate(chunks):
        print(f"--- Chunk {i} ({len(c)} chars) ---")
        print(c)
        print()