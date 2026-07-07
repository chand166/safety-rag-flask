"""
配置加载模块 — 从 YAML 读取配置，解析相对路径，提供全局 Config 对象
"""
import os
from pathlib import Path
from typing import Dict

# 项目根目录（此文件位于 src/config_loader.py，项目根在其 parent 目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CONFIG_CACHE = None
_CONFIG_PATH = None


def get_config_path() -> Path:
    """返回配置文件路径（优先环境变量，默认项目路径）"""
    env_path = os.environ.get("SAFETY_RAG_CONFIG")
    if env_path:
        return Path(env_path)
    return _PROJECT_ROOT / "config" / "config.yaml"


def _resolve_path(path_str: str, config_file_dir: Path = None) -> str:
    """
    解析路径：
    - 相对路径（以 ./ 开头）→ 基于项目根目录或配置文件所在目录
    - 绝对路径（Windows 如 D:\\xxx 或 Linux 如 /home/xxx）→ 原样返回
    """
    if not path_str:
        return path_str

    path = Path(path_str)

    # 已是绝对路径
    if path.is_absolute():
        return str(path)

    # 相对路径 → 基于项目根目录
    return str((_PROJECT_ROOT / path).resolve())


def _resolve_all_paths(cfg: dict, config_file_dir: Path) -> dict:
    """递归解析配置中所有路径字段"""
    path_fields = {
        "knowledge_base",
        "persist_directory",
    }
    for key, value in cfg.items():
        if key in path_fields and isinstance(value, str):
            cfg[key] = _resolve_path(value, config_file_dir)
        elif isinstance(value, dict):
            cfg[key] = _resolve_all_paths(value, config_file_dir)
    return cfg


def load_config(reload: bool = False) -> Dict:
    """加载配置（带缓存），自动解析相对路径"""
    global _CONFIG_CACHE, _CONFIG_PATH

    if _CONFIG_CACHE is not None and not reload:
        return _CONFIG_CACHE

    import yaml

    cfg_path = get_config_path()
    _CONFIG_PATH = cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f)

    # 解析相对路径为绝对路径
    _CONFIG_CACHE = _resolve_all_paths(_CONFIG_CACHE, cfg_path.parent)

    return _CONFIG_CACHE


config = load_config()


def get_project_root() -> Path:
    """返回项目根目录"""
    return _PROJECT_ROOT