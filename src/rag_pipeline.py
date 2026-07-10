"""
RAG 管线 — 混合搜索（精确匹配 + 语义检索）
"""
import os
import re
from typing import List, Dict, Optional
from pathlib import Path

from src.vector_store import search as vector_search
from src.llm_client import get_llm
from src.config_loader import config

SYSTEM_PROMPT = """你是一个专业的安全培训知识库助手，基于提供的文档内容回答用户问题。

回答规则：
1. 直接输出最终答案，不要输出任何思考过程、分析过程或"让我"、"首先"、"根据"、"用户询问的是"等前缀
2. 严格基于文档内容，不要编造信息
3. 如果文档不足以回答，直接说"文档中没有找到相关信息"
4. 用中文回答，简洁明了
5. 在回答末尾另起一行列出"来源：文件名"

输出格式：
答案：[你的回答]

来源：文件名"""


# ============================================================
# 查询扩展（缩写 → 全称）
# ============================================================

_QUERY_EXPANSIONS = {
    "gbl": "γ-丁内酯 丁内酯 gamma-丁内酯",
    "dmf": "二甲基甲酰胺",
    "dmso": "二甲基亚砜",
    "thf": "四氢呋喃",
    "meoh": "甲醇",
    "etoh": "乙醇",
    "ipa": "异丙醇",
    "mek": "丁酮",
    "tfa": "三氟乙酸",
    "pma": "磷钼酸",
    "tms": "四甲基硅烷",
    "pma": "磷钼酸",
    "nmp": "n-甲基吡咯烷酮",
    "dmac": "二甲基乙酰胺",
    "dcm": "二氯甲烷",
    "ea": "乙酸乙酯",
    "acn": "乙腈",
    "pce": "四氯乙烯",
    "tce": "三氯乙烯",
    "sds": "十二烷基硫酸钠",
}


def _expand_query(query: str) -> str:
    """扩展查询中的缩写为全称/别名，提高检索命中率。"""
    query_lower = query.lower()
    expansions = []
    for abbr, full in _QUERY_EXPANSIONS.items():
        if abbr in query_lower:
            expansions.append(full)
    if expansions:
        return query + " " + " ".join(expansions)
    return query


# ============================================================
# 精确匹配搜索
# ============================================================

def exact_search(query: str, max_files: int = 5) -> List[Dict]:
    """对 ChromaDB 全部文本进行关键词匹配，自动提取关键词。"""
    try:
        from src.vector_store import get_client, get_collection
        client = get_client()
        collection = get_collection(client)
        all_data = collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    if not all_data or not all_data["documents"]:
        return []

    # 提取关键词
    stop_words = {"的", "是", "了", "在", "有", "和", "就", "不", "都", "一", "个", "上", "也",
                  "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己",
                  "这", "他", "她", "它", "们", "什么", "怎么", "多久", "哪些", "哪里", "如何",
                  "为什么", "吗", "呢", "啊", "吧", "呀", "嘛", "请问", "我想", "知道", "了解",
                  "介绍", "说明", "告诉", "请", "为", "我", "你", "能", "可以", "需要", "应该",
                  "必须", "是否", "有没有", "多少", "把", "被", "让", "给", "对", "向"}

    segments = re.split(r'[，。！？、；：\s,?!.\-()（）]', query)
    keywords = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) < 2:
            continue
        parts = re.split(r'[的的是了和在就于]', seg)
        for p in parts:
            p = p.strip()
            if len(p) >= 2 and p not in stop_words:
                keywords.append(p)
    keywords = list(dict.fromkeys(keywords))
    if not keywords:
        keywords = [query]

    matches = []
    for i, doc_text in enumerate(all_data["documents"]):
        if not doc_text:
            continue
        text_lower = doc_text.lower()
        matched = [kw for kw in keywords if kw.lower() in text_lower]
        if not matched:
            continue
        meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
        first_kw = matched[0]
        pos = text_lower.find(first_kw.lower())
        start = max(0, pos - 80)
        end = min(len(doc_text), pos + len(first_kw) + 160)
        snippet = doc_text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(doc_text):
            snippet = snippet + "..."
        matches.append({
            "filepath": meta.get("filepath", ""),
            "filename": meta.get("filename", "未知"),
            "relpath": meta.get("relpath", ""),
            "category": meta.get("category", "未分类"),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_total": meta.get("chunk_total", 1),
            "score": 0.0,
            "snippet": snippet,
        })

    seen = set()
    unique = []
    for m in matches:
        key = m["relpath"]
        if key not in seen:
            seen.add(key)
            unique.append(m)
        if len(unique) >= max_files:
            break
    return unique


# ============================================================
# 清洗回答（去掉推理过程）
# ============================================================

_THINK_PREFIXES = (
    "用户询问的是", "用户的问题是", "用户问的是",
    "让我逐一查看", "让我先查看", "让我看看", "让我来",
    "这是一个关于", "这个问题是",
    "好的，", "好的,",
    "首先，", "首先,",
    "根据文档", "根据提供的",
    "我需要", "我来",
    "综合各文档", "综合来看",
    "所以，", "所以,",
    "因此，", "因此,",
    "在回答这个问题之前",
    "回答：", "回答:",
)


def _clean_answer(text: str) -> str:
    """去掉 LLM 输出的推理过程，只保留最终回答。"""
    original = text

    # 1. 处理各种推理标签格式
    # MiniMax: think...<response>...
    if '<response>' in text:
        text = re.sub(r'.*?<response>\s*', '', text, flags=re.DOTALL).strip()
    # 带标签的 think 块
    text = re.sub(r'.*?think>\s*', '', text, flags=re.DOTALL).strip()
    # 开头的 "thinking" 无标签
    text = re.sub(r'^thinking\s*', '', text, flags=re.IGNORECASE).strip()

    # 2. 查找 "答案：" 或 "回答：" 作为最终回答的分隔符
    markers = ['答案：', '答案:', '回答：', '回答:', '答：', '答:']
    for m in markers:
        if m in text:
            # 取最后一个出现的位置（避免前面的思考内容中的"答案"）
            idx = text.rfind(m)
            after = text[idx + len(m):].strip()
            if len(after) > 10:  # 确保有实际内容
                text = after
                break

    # 3. 如果文本还很⻓（>200字）且包含思考痕迹，进一步清理
    if len(text) > 200:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        # 找到第一个看起来像实际回答的行
        thinking_patterns = re.compile(
            r'^(用户|让我|这[是个]|从文档|在文档|查看|搜索|基于|首先|第一|'
            r'综合|所以|因此|综上|回答|答案|文档中|文档1|文档2|文档[0-9]|'
            r'虽然|不过|严格|实际|让我再|好的|好[的，]|是的[，，]|'
            r'根据.*文档|从.*来看|在.*中|第[一二三]步|接下来|最后)'
        )
        filtered = []
        for line in lines:
            if not filtered and thinking_patterns.match(line):
                continue  # 跳过开头的思考行
            filtered.append(line)
        if filtered:
            text = '\n'.join(filtered)

    # 4. 去掉开头残留的"答案：""回答："等前缀
    text = re.sub(r'^[答案回答答][：:]\s*', '', text).strip()

    # 5. 去掉常见的思考前缀
    for prefix in _THINK_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    # 6. 如果内容太短，回退到原始文本
    if len(text) < 20 and len(original) > 50:
        text = original

    # 5. 检查是否还有思考内容残留：如果内容以 "XX是"、"XX指" 的句式开头
    #    且后面跟了很长的解释，这是正常的。保留。

    # 6. 如果文本太长（>300字）且包含 "1."、"2." 这样的编号，
    #    尝试去掉开头的"总结性"前言
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(text) > 150 and len(lines) >= 3:
        # 检查第1行是不是"总结性陈述"而不是实际内容
        first = lines[0]
        if re.match(r'^[这那该本]', first) and len(first) < 30:
            # 可能是过渡句，去掉
            text = '\n'.join(lines[1:]).strip()
            lines = [l.strip() for l in text.split('\n') if l.strip()]

    # 7. 去掉首尾多余的空白行
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


# ============================================================
# 主回答函数
# ============================================================

def generate_answer(question: str, top_k: int = 5, show_context: bool = False,
                    model: str = None, web_search_enabled: bool = True,
                    conversation_history: list = None) -> Dict:
    """生成 RAG 回答（混合搜索）。
    
    搜索策略（保护本地数据隐私）：
    1. 优先本地知识库检索（向量 + 精确匹配）
    2. 本地有结果 → 直接返回，绝不联网
    3. 本地无结果 + web_search_enabled=True → 联网搜索
    4. 本地无结果 + web_search_enabled=False → 模型自身知识
    
    注意：联网搜索时只发送用户问题，绝不发送本地文档内容，避免数据泄露。
    """
    llm = get_llm(model_key=model)
    system_prompt = SYSTEM_PROMPT + "\n\n重要：只输出最终答案，严禁输出任何思考过程。"

    # ---- 对话历史（用于上下文理解） ----
    history_messages = []
    if conversation_history:
        for msg in conversation_history[-6:]:
            role = "assistant" if msg["role"] == "assistant" else "user"
            history_messages.append({"role": role, "content": msg["content"]})

    # ---- 0. 查询扩展（缩写 → 全称，提高检索命中率） ----
    search_query = _expand_query(question)

    # ---- 1. 语义检索 ----
    vector_results = vector_search(search_query, top_k=top_k)
    has_relevant = any(r["score"] < 0.45 for r in vector_results) if vector_results else False

    # ---- 2. 精确匹配（无论向量结果如何都执行） ----
    exact_results = exact_search(search_query, max_files=top_k)

    # ---- 3. 合并检索结果（精确匹配优先，无精确匹配时尝试联网） ----
    all_docs = []
    seen_paths = set()

    # 精确匹配有命中 → 本地文档可信，直接用
    if exact_results:
        all_docs = exact_results
        seen_paths = {r["relpath"] for r in exact_results}
        # 补充向量结果中未覆盖的文档
        if vector_results and has_relevant:
            for r in vector_results:
                if r["relpath"] not in seen_paths:
                    all_docs.append(r)
                    seen_paths.add(r["relpath"])
    elif vector_results and has_relevant:
        # 只有语义匹配，没有精确匹配 → 文档可能只是沾边
        # 如果联网搜索开启，先尝试联网搜索
        if web_search_enabled:
            from src.web_search import web_search, format_search_context
            web_results = web_search(question, max_results=5)
            if web_results and "error" not in web_results[0]:
                context = format_search_context(web_results)
                system_prompt = "你是一个知识库AI助手。以下是从互联网搜索到的相关信息，请基于这些信息回答用户的问题。如果信息不足，可以补充你的知识。用中文回答，简洁明了。"
                user_msg = f"互联网搜索结果：\n{context}\n\n用户问题：{question}"
                answer = llm.chat([
                    {"role": "system", "content": system_prompt},
                ] + history_messages + [
                    {"role": "user", "content": user_msg},
                ])
                answer = _clean_answer(answer)
                sources = []
                for r in web_results:
                    sources.append({
                        "filename": r["title"],
                        "relpath": r["url"],
                        "filepath": r["url"],
                        "category": "网络搜索",
                        "score": 0,
                        "snippet": r["snippet"],
                    })
                return {"question": question, "answer": answer, "sources": sources,
                        "from_kb": False, "match_type": "web_search"}
        # 联网搜索未开启或失败 → 用本地文档（虽然不精确）
        all_docs = list(vector_results)
        seen_paths = {r["relpath"] for r in vector_results}

    if all_docs:
        # 构建上下文
        context_parts = []
        for i, r in enumerate(all_docs, 1):
            text = r.get("chunk_text") or r.get("snippet", "")
            context_parts.append(
                f"[文档 {i}] 来源: {r['filename']}\n"
                f"分类: {r['category']}\n"
                f"内容:\n{text}\n"
            )
        context = "\n---\n".join(context_parts)
        user_msg = f"请基于以下文档内容回答用户的问题。\n\n文档内容：\n{context}\n\n用户问题：{question}"
        answer = llm.chat([
            {"role": "system", "content": system_prompt},
        ] + history_messages + [
            {"role": "user", "content": user_msg},
        ])
        answer = _clean_answer(answer)

        sources = []
        seen = set()
        for r in all_docs:
            key = r["relpath"]
            if key not in seen:
                seen.add(key)
            sources.append({
                "filename": r["filename"], "relpath": r["relpath"],
                "filepath": r["filepath"], "category": r["category"],
                "score": round(r.get("score", 0), 4),
                "snippet": (r.get("chunk_text") or r.get("snippet", ""))[:150] + "...",
            })
        return {"question": question, "answer": answer, "sources": sources,
                "from_kb": True, "match_type": "vector"}

    # ---- 2. 知识库无结果 — 网络搜索补充（仅在开启时） ----
    if web_search_enabled:
        from src.web_search import web_search, format_search_context
        web_results = web_search(question, max_results=5)
        if web_results and "error" not in web_results[0]:
            context = format_search_context(web_results)
            system_prompt = "你是一个知识库AI助手。以下是从互联网搜索到的相关信息，请基于这些信息回答用户的问题。如果信息不足，可以补充你的知识。用中文回答，简洁明了。"
            user_msg = f"互联网搜索结果：\n{context}\n\n用户问题：{question}"
            answer = llm.chat([
                {"role": "system", "content": system_prompt},
            ] + history_messages + [
                {"role": "user", "content": user_msg},
            ])
            answer = _clean_answer(answer)
            sources = []
            for r in web_results:
                sources.append({
                    "filename": r["title"],
                    "relpath": r["url"],
                    "filepath": r["url"],
                    "category": "网络搜索",
                    "score": 0,
                    "snippet": r["snippet"],
                })
            return {"question": question, "answer": answer, "sources": sources,
                    "from_kb": False, "match_type": "web_search"}

    # ---- 3. 知识库无结果，联网搜索未开启或失败 — LLM 自身知识 ----
    note = ""
    if not web_search_enabled:
        note = "（联网搜索已关闭）"
    answer = llm.chat([
        {"role": "system", "content": "你是一个安全培训助手。直接给出最终答案，不要输出思考过程。"},
    ] + history_messages + [
        {"role": "user", "content": f"用户问题：{question}\n\n知识库中未检索到相关文档，请基于你的知识回答。"},
    ])
    answer = _clean_answer(answer)
    return {"question": question, "answer": f"{answer}\n\n⚠️ 说明：知识库中未检索到相关文档{note}，以上回答基于模型自身知识。",
            "sources": [], "from_kb": False, "match_type": "fallback"}