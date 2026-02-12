"""
配置管理模块 (Configuration Manager)

提供统一的配置加载和访问接口，支持：
- 从 YAML 文件加载配置
- 从环境变量覆盖配置
- 点号路径访问嵌套配置（如 config.get("api.embedding.model")）
- 默认值回退
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 默认配置
DEFAULT_CONFIG = {
    "system": {
        "n_experts": None,  # None 表示自动识别
        "expert_identification_method": "clustering",  # 'clustering', 'lda', 'entity_based'
        "random_seed": 42,
        "log_level": "INFO",
    },
    "api": {
        "embedding": {
            "api_key": "sk-xxx",
            "base_url": "https://api.openai.com/v1",
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "batch_size": 64,
            "max_retries": 3,
            "timeout": 60,
        },
        "chat": {
            "api_key": "sk-xxx",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 512,
            "max_retries": 3,
            "timeout": 60,
        },
    },
    "retrieval": {
        "vector_top_k": 5,
        "rerank": True,
        "hybrid_alpha": 0.7,  # 向量检索权重（1-alpha 为图谱检索权重）
        "similarity_threshold": 0.3,
    },
    "reasoning": {
        "max_hops": 5,
        "max_iterations": 3,
        "reasoning_mode": "chain",  # 'chain', 'graph', 'iterative'
        "confidence_threshold": 0.5,
    },
    "storage": {
        "kb_base_path": "./data/knowledge_bases",
        "raw_data_path": "./data/raw",
        "processed_data_path": "./data/processed",
        "cache_enabled": True,
        "cache_path": "./data/cache",
    },
    "vector_index": {
        "index_type": "faiss",  # 'faiss', 'annoy', 'simple'
        "normalize_embeddings": True,
        "nprobe": 10,  # FAISS IVF 参数
    },
    "graph": {
        "max_path_length": 3,
        "min_confidence": 0.5,
    },
}


class Config:
    """
    配置管理器

    支持从 YAML 文件加载配置，提供点号路径访问方式。

    Usage:
        >>> config = Config.load("config.yaml")
        >>> model_name = config.get("api.embedding.model")
        >>> top_k = config.get("retrieval.vector_top_k", default=5)
    """

    _instance: Optional[Config] = None  # 单例实例

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化配置

        Args:
            config_dict: 配置字典，如果为 None 则使用默认配置
        """
        self._config = self._deep_merge(DEFAULT_CONFIG, config_dict or {})

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> Config:
        """
        从 YAML 文件加载配置

        Args:
            config_path: 配置文件路径。如果为 None，尝试加载项目根目录的 config.yaml

        Returns:
            Config 实例
        """
        if config_path is None:
            config_path = str(PROJECT_ROOT / "config.yaml")

        config_dict = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    config_dict = loaded
                logger.info(f"配置已从 {config_path} 加载")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}，使用默认配置")
        else:
            logger.info(f"配置文件 {config_path} 不存在，使用默认配置")

        instance = cls(config_dict)
        cls._instance = instance
        return instance

    @classmethod
    def get_instance(cls) -> Config:
        """
        获取单例实例

        Returns:
            Config 单例实例，如果未初始化则使用默认配置创建
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        通过点号路径获取配置值

        Args:
            key_path: 用点号分隔的键路径，如 "api.embedding.model"
            default: 默认值，键不存在时返回

        Returns:
            配置值

        Example:
            >>> config.get("api.embedding.model")
            'text-embedding-3-small'
            >>> config.get("retrieval.vector_top_k", 10)
            5
        """
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any):
        """
        设置配置值

        Args:
            key_path: 用点号分隔的键路径
            value: 要设置的值
        """
        keys = key_path.split(".")
        config = self._config
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取一个完整的配置节

        Args:
            section: 节名称

        Returns:
            配置节字典，不存在返回空字典
        """
        return self._config.get(section, {})

    @property
    def all(self) -> Dict[str, Any]:
        """返回完整配置字典的副本"""
        return dict(self._config)

    def save(self, config_path: str):
        """
        保存配置到 YAML 文件

        Args:
            config_path: 保存路径
        """
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self._config, f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        logger.info(f"配置已保存到: {config_path}")

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """
        深度合并两个字典

        override 中的值会覆盖 base 中的同名键。

        Args:
            base: 基础字典
            override: 覆盖字典

        Returns:
            合并后的新字典
        """
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        sections = list(self._config.keys())
        return f"Config(sections={sections})"
