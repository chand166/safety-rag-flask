"""OCR 辅助脚本 — 被 document_loader 通过 subprocess 调用
用法: python3 ocr_worker.py <图片路径>
输出: stdout 只有一行 JSON，PaddleOCR 日志全部重定向到 stderr
"""
import sys, json, os

# 所有 PaddleOCR 日志输出到 stderr
sys.stdout = os.fdopen(os.dup(1), 'w')
_real_stdout = sys.stdout
sys.stderr = os.fdopen(os.dup(2), 'w')

from paddleocr import PaddleOCR

# 初始化时所有 print 输出到 stderr
import contextlib

@contextlib.contextmanager
def _stdout_to_stderr():
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old

with _stdout_to_stderr():
    ocr = PaddleOCR(lang="ch", use_angle_cls=True)

def ocr_image(img_path: str) -> str:
    with _stdout_to_stderr():
        result = ocr.ocr(img_path, cls=True)
    texts = []
    for group in result:
        for line in group:
            texts.append(line[1][0])
    return "\n".join(texts)

if __name__ == "__main__":
    try:
        text = ocr_image(sys.argv[1])
        _real_stdout.write(json.dumps({"success": True, "text": text}, ensure_ascii=False) + "\n")
        _real_stdout.flush()
    except Exception as e:
        _real_stdout.write(json.dumps({"success": False, "error": str(e)}, ensure_ascii=False) + "\n")
        _real_stdout.flush()