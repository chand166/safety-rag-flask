"""
网络搜索模块 — 知识库无结果时从网络补充信息

通过 Bing 搜索获取网页摘要，用于补充 LLM 回答的上下文。
"""
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
_BING_URL = "https://www.bing.com/search"


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """通过 Bing 搜索，返回标题+链接+摘要。"""

    # 优化搜索关键词：去掉"什么是"、"如何"等疑问前缀，保留核心词
    search_q = query.strip()
    # 去掉常见疑问前缀
    for prefix in ["什么是", "什么是", "如何", "怎么", "怎样", "哪些", "哪里", "为什么"]:
        if search_q.startswith(prefix):
            search_q = search_q[len(prefix):].strip()
            break
    # 如果去掉前缀后太短，用原词
    if len(search_q) < 2:
        search_q = query.strip()

    try:
        r = requests.get(
            _BING_URL,
            params={"q": search_q, "cc": "cn", "mkt": "zh-CN"},
            headers=_SEARCH_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        return [{"error": f"搜索请求失败: {e}"}]

    soup = BeautifulSoup(r.text, "html.parser")
    results = []

    for item in soup.select(".b_algo")[:max_results]:
        link_tag = item.select_one("h2 a")
        snippet_tag = item.select_one(".b_caption p")
        if not link_tag:
            continue
        title = link_tag.get_text(strip=True)
        href = link_tag.get("href", "")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        results.append({
            "title": title,
            "url": href,
            "snippet": snippet,
        })

    if not results:
        # 容错：尝试备选 CSS 选择器
        for item in soup.select("li.b_algo")[:max_results]:
            link_tag = item.select_one("h2 a")
            snippet_tag = item.select_one(".b_caption p")
            if not link_tag:
                continue
            results.append({
                "title": link_tag.get_text(strip=True),
                "url": link_tag.get("href", ""),
                "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
            })

    # 如果结果都不相关（标题不含核心词或全是日文），用引号搜一次
    if results and search_q != query.strip():
        core_words = search_q[:2]
        has_relevant = any(core_words in r["title"] for r in results)
        if not has_relevant:
            # 后备：用引号括起原词搜索
            try:
                r2 = requests.get(
                    _BING_URL,
                    params={"q": f'"{query.strip()}"', "cc": "cn", "mkt": "zh-CN"},
                    headers=_SEARCH_HEADERS,
                    timeout=10,
                )
                soup2 = BeautifulSoup(r2.text, "html.parser")
                results2 = []
                for item in soup2.select(".b_algo")[:max_results]:
                    link_tag = item.select_one("h2 a")
                    snippet_tag = item.select_one(".b_caption p")
                    if not link_tag:
                        continue
                    results2.append({
                        "title": link_tag.get_text(strip=True),
                        "url": link_tag.get("href", ""),
                        "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
                    })
                if results2:
                    results = results2
            except Exception:
                pass

    return results


def format_search_context(results: List[Dict]) -> str:
    """把搜索结果格式化为上下文文本。"""
    if not results:
        return ""
    if "error" in results[0]:
        return ""  # 搜索失败，不提供上下文

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[搜索结果 {i}]\n"
            f"标题: {r['title']}\n"
            f"链接: {r['url']}\n"
            f"摘要: {r['snippet']}\n"
        )
    return "\n---\n".join(parts)


if __name__ == "__main__":
    results = web_search("碳中和是什么意思")
    print(f"找到 {len(results)} 条结果:\n")
    for r in results:
        print(f"  📄 {r['title']}")
        print(f"     {r['url']}")
        print(f"     {r['snippet'][:100]}...")
        print()