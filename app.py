# 安全培训 RAG 知识库智能体 — Flask 应用
# WSGI 入口: gunicorn app:app
import os
import sys
import json
import base64
import sqlite3
from pathlib import Path
from datetime import datetime
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
# 历史记录数据库
# ============================================================
DB_PATH = _PROJECT_ROOT / "data" / "chat_history.db"

def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    return db

def init_db():
    os.makedirs(str(DB_PATH.parent), exist_ok=True)
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT '新对话',
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT DEFAULT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)
    """)
    db.commit()
    db.close()

init_db()


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
# 问答 API（自动保存历史记录）
# ============================================================
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "请提供问题"}), 400

    question = data["question"].strip()
    conversation_id = data.get("conversation_id")

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    top_k = data.get("top_k", 5)
    model = data.get("model")
    web_search_enabled = data.get("web_search_enabled", True)

    try:
        # 获取对话历史（最近6条消息，用于上下文理解）
        history = []
        if conversation_id:
            db = get_db()
            rows = db.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 6",
                (conversation_id,)
            ).fetchall()
            db.close()
            for r in reversed(rows):
                history.append({"role": r["role"], "content": r["content"]})

        result = generate_answer(question, top_k=top_k, model=model,
                                  web_search_enabled=web_search_enabled,
                                  conversation_history=history)

        # ===== 保存到历史记录 =====
        db = get_db()
        if not conversation_id:
            cursor = db.execute(
                "INSERT INTO conversations (title) VALUES (?)",
                (question[:50],)
            )
            conversation_id = cursor.lastrowid
        else:
            conv = db.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()
            if conv and conv["title"] == "新对话":
                db.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (question[:50], conversation_id)
                )

        db.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, 'user', ?)",
            (conversation_id, question)
        )
        db.execute(
            "INSERT INTO messages (conversation_id, role, content, sources) VALUES (?, 'assistant', ?, ?)",
            (conversation_id, result["answer"], json.dumps(result.get("sources", []), ensure_ascii=False))
        )
        db.execute(
            "UPDATE conversations SET updated_at = datetime('now','localtime') WHERE id = ?",
            (conversation_id,)
        )
        db.commit()
        db.close()

        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "match_type": result.get("match_type", "vector"),
            "from_kb": result.get("from_kb", True),
            "conversation_id": conversation_id,
        })
    except Exception as e:
            return jsonify({"error": str(e)}), 500


# ============================================================
# 历史记录 API
# ============================================================
@app.route("/history")
def list_conversations():
    db = get_db()
    rows = db.execute("""
        SELECT c.id, c.title, c.created_at, c.updated_at,
               (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as msg_count,
               COALESCE(
                   (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY id LIMIT 1),
                   ''
               ) as preview
        FROM conversations c
        ORDER BY c.updated_at DESC
        LIMIT 50
    """).fetchall()
    db.close()
    return jsonify([{
        "id": r["id"],
        "title": r["title"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "msg_count": r["msg_count"],
        "preview": r["preview"][:80] if r["preview"] else ""
    } for r in rows])


@app.route("/history/<int:conv_id>")
def get_conversation(conv_id):
    db = get_db()
    conv = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    if not conv:
        db.close()
        return jsonify({"error": "会话不存在"}), 404
    messages = db.execute(
        "SELECT id, role, content, sources, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
        (conv_id,)
    ).fetchall()
    db.close()
    return jsonify({
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "messages": [{
            "id": m["id"],
            "role": m["role"],
            "content": m["content"],
            "sources": json.loads(m["sources"]) if m["sources"] else None,
            "created_at": m["created_at"],
        } for m in messages]
    })


@app.route("/history/new", methods=["POST"])
def new_conversation():
    db = get_db()
    cursor = db.execute("INSERT INTO conversations (title) VALUES ('新对话')")
    conv_id = cursor.lastrowid
    db.commit()
    db.close()
    return jsonify({"id": conv_id, "title": "新对话", "messages": []})


@app.route("/history/rename", methods=["POST"])
def rename_conversation():
    data = request.get_json()
    if not data or "id" not in data or "title" not in data:
        return jsonify({"error": "参数不完整"}), 400
    db = get_db()
    db.execute("UPDATE conversations SET title = ? WHERE id = ?",
               (data["title"].strip()[:100], data["id"]))
    db.commit()
    db.close()
    return jsonify({"status": "ok"})


@app.route("/history/<int:conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    db = get_db()
    db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    db.commit()
    db.close()
    return jsonify({"status": "ok"})


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """删除所有对话"""
    db = get_db()
    db.execute("DELETE FROM messages")
    db.execute("DELETE FROM conversations")
    db.commit()
    db.close()
    return jsonify({"status": "ok"})


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
