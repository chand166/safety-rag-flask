# 安全培训 RAG 知识库智能体 — Flask 应用
# WSGI 入口: gunicorn app:app
import os
import sys
import base64
from pathlib import Path
from functools import wraps

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag_pipeline import generate_answer
from src.vector_store import get_stats
from src.config_loader import config
from src.llm_client import get_available_models

app = Flask(__name__)

# 基础配置
app.config["SECRET_KEY"] = "safety-rag-secret-key-change-in-production"
app.config["JSON_AS_ASCII"] = False


# ============================================================
# 首页 — 聊天界面
# ============================================================
@app.route("/")
def index():
    try:
        stats = get_stats()
        kb_info = {
            "chunks": stats["total_chunks"],
            "categories": stats["categories"],
        }
    except Exception:
        kb_info = {"chunks": 0, "categories": []}
    return render_template("index.html", kb_info=kb_info)


# ============================================================
# 问答 API
# ============================================================
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "请提供问题"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    top_k = data.get("top_k", 5)
    model = data.get("model")

    try:
        result = generate_answer(question, top_k=top_k, model=model)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "match_type": result.get("match_type", "vector"),
            "from_kb": result.get("from_kb", True),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 知识库统计
# ============================================================
@app.route("/stats")
def stats():
    try:
        s = get_stats()
        return jsonify({
            "chunks": s["total_chunks"],
            "categories": s["categories"],
            "category_count": s["category_count"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 文件列表
# ============================================================
@app.route("/files")
def file_list():
    try:
        from src.document_loader import scan_documents
        docs = scan_documents()
        return jsonify({"files": [{"filename": d["filename"], "filepath": d["filepath"], "category": d["category"]} for d in docs]})
    except Exception as e:
        return jsonify({"files": [], "error": str(e)})


# ============================================================
# 重建索引
# ============================================================
@app.route("/rebuild", methods=["POST"])
def rebuild():
    try:
        from src.document_loader import load_all_documents
        from src.text_splitter import split_documents
        from src.vector_store import index_documents
        docs = load_all_documents(verbose=False)
        chunks = split_documents(docs)
        count = index_documents(chunks)
        return jsonify({"status": "ok", "count": count})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ============================================================
# 打开文件 API
# ============================================================
@app.route("/open-file", methods=["POST"])
def open_file():
    data = request.get_json()
    if not data or "path" not in data:
        return jsonify({"error": "请提供文件路径"}), 400
    filepath = data["path"]
    import platform
    if platform.system() == "Windows":
        try:
            os.startfile(filepath)
            return jsonify({"status": "ok", "message": f"已打开: {filepath}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "服务器环境不支持直接打开文件"}), 400


# ============================================================
# 下载文件（浏览器直接打开/下载）
# ============================================================
@app.route("/download-file")
def download_file():
    filepath = request.args.get("path", "")
    if not filepath or not os.path.exists(filepath):
        return "文件不存在", 404
    filename = os.path.basename(filepath)
    return send_file(filepath, as_attachment=True, download_name=filename)


# ============================================================
# 可用模型列表
# ============================================================
@app.route("/models")
def list_models():
    return jsonify({"models": get_available_models()})


# ============================================================
# 健康检查
# ============================================================
@app.route("/health")
def health():
    return "ok"


# ============================================================
# 静态文件
# ============================================================
@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(str(_PROJECT_ROOT / "assets"), filename)


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    port = config.get("server", {}).get("port", 8501)
    host = config.get("server", {}).get("host", "0.0.0.0")
    print(f"启动 Flask 开发服务器: http://{host}:{port}")
    app.run(host=host, port=port, debug=True)