#!/bin/bash
set -e

echo "========================================"
echo "  知识库AI — Flask + Gunicorn 部署"
echo "========================================"

echo ""
echo "📋 1. 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ 已创建"
fi

echo ""
echo "📋 2. 安装依赖..."
. venv/bin/activate
pip install -r requirements.txt -q
echo "  ✓ 完成"

echo ""
echo "📋 3. 检查知识库..."
if [ ! -d "knowledge_base" ] || [ -z "$(ls -A knowledge_base/ 2>/dev/null)" ]; then
    echo "  ⚠️ 知识库为空，请放入文件后运行 python scripts/index_all.py"
    mkdir -p knowledge_base
else
    echo "  ✓ $(find knowledge_base -type f | wc -l) 个文件"
fi

echo ""
echo "📋 4. 索引知识库..."
if [ -d "knowledge_base" ] && [ "$(find knowledge_base -type f | wc -l)" -gt 0 ]; then
    . venv/bin/activate
    python scripts/index_all.py
    echo "  ✓ 索引完成"
fi

echo ""
echo "========================================"
echo "  ✅ 部署准备完成！"
echo "========================================"
echo ""
echo "启动方式:"
echo ""
echo "  # 开发模式（测试用）"
echo "  . venv/bin/activate"
echo "  python app.py"
echo "  访问 http://localhost:8501"
echo ""
echo "  # 生产模式（Gunicorn）"
echo "  . venv/bin/activate"
echo "  gunicorn -w 4 -b 0.0.0.0:8501 app:app"
echo ""
echo "  # 后台运行"
echo "  nohup venv/bin/gunicorn -w 4 -b 0.0.0.0:8501 app:app > app.log 2>&1 &"
echo ""
echo "  # 停止"
echo "  pkill -f 'gunicorn.*app:app'"
echo ""