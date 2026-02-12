"""
向量索引构建器 (Vector Index Builder)

负责将文本文档转换为向量表示，并构建高效的向量检索索引。

核心流程：
1. 通过 OpenAI 兼容 API 调用 Embedding 模型
2. 文本批量向量化
3. 构建 FAISS 索引（支持快速相似度搜索）
4. 保存/加载索引和元数据

所有 Embedding 计算均通过远程 API 完成，无需本地 GPU 或模型文件。
支持 OpenAI 官方、Azure OpenAI、以及任何兼容接口（vLLM、Ollama、LiteLLM 等）。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from data.models import Document
from kb_builder.models import SearchResult, VectorIndex
from utils.config import Config

logger = logging.getLogger(__name__)


class VectorIndexBuilder:
    """
    向量索引构建器

    通过 OpenAI 兼容 API 将文本向量化并构建检索索引，支持：
    - 任意 OpenAI 兼容的 Embedding API
    - 自动批量处理与重试
    - FAISS / numpy 索引

    Usage:
        >>> builder = VectorIndexBuilder()
        >>> documents = ["Apple was founded by Steve Jobs.", "Paris is the capital of France."]
        >>> metadata = [{"title": "Apple"}, {"title": "France"}]
        >>> index = builder.build_index(documents, metadata)
        >>> results = index.search(builder.encode("Who founded Apple?"), top_k=3)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
    ):
        """
        初始化向量索引构建器

        Args:
            config: 配置对象（如果提供，将从中读取 API 参数）
        """
        # 加载配置
        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self._config = config

        # 从配置读取 Embedding API 参数
        self.api_key = config.get("api.embedding.api_key", "sk-xxx")
        self.base_url = config.get("api.embedding.base_url", "https://api.openai.com/v1")
        self.model = config.get("api.embedding.model", "text-embedding-3-small")
        self.dimension = config.get("api.embedding.dimension", 1536)
        self.batch_size = config.get("api.embedding.batch_size", 64)
        self.max_retries = config.get("api.embedding.max_retries", 3)
        self.timeout = config.get("api.embedding.timeout", 60)

        self.index_type = config.get("vector_index.index_type", "faiss")

        # OpenAI 客户端（延迟初始化）
        self._client = None

        logger.info(
            f"VectorIndexBuilder 初始化: model={self.model}, "
            f"base_url={self.base_url}, dimension={self.dimension}"
        )

    # ==========================================
    # API 客户端
    # ==========================================

    def _get_client(self):
        """
        获取 OpenAI 客户端（延迟初始化）

        Returns:
            openai.OpenAI 客户端实例
        """
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            logger.info(f"OpenAI Embedding 客户端已初始化: {self.base_url}")

        return self._client

    # ==========================================
    # 向量编码
    # ==========================================

    def encode(self, texts: str | List[str],
               show_progress: bool = False) -> np.ndarray:
        """
        将文本编码为向量（通过 API 调用）

        Args:
            texts: 单个文本或文本列表
            show_progress: 是否显示进度条

        Returns:
            向量数组，shape=(n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        # 分批调用 API
        all_embeddings = []
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        iterator = batches
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(batches, desc="Embedding API 调用")
            except ImportError:
                pass

        client = self._get_client()

        for batch_texts in iterator:
            # 清理文本：去除空字符串，截断过长文本
            cleaned = [t.strip()[:8000] if t.strip() else " " for t in batch_texts]

            try:
                response = client.embeddings.create(
                    input=cleaned,
                    model=self.model,
                )

                batch_embeddings = [
                    item.embedding for item in response.data
                ]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Embedding API 调用失败: {e}")
                # 出错时填充零向量，确保维度对齐
                for _ in cleaned:
                    all_embeddings.append([0.0] * self.dimension)

        result = np.array(all_embeddings, dtype=np.float32)

        # 归一化
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        result = result / norms

        return result

    # ==========================================
    # 索引构建
    # ==========================================

    def build_index(self, documents: List[str],
                    metadata: Optional[List[Dict]] = None,
                    show_progress: bool = True) -> VectorIndex:
        """
        构建向量索引

        完整流程：
        1. 通过 API 进行文本向量化
        2. 构建索引（FAISS 或 numpy）
        3. 关联元数据

        Args:
            documents: 文档文本列表
            metadata: 每个文档的元数据列表（可选）
            show_progress: 是否显示进度

        Returns:
            构建好的 VectorIndex 实例
        """
        if not documents:
            logger.warning("文档列表为空，返回空索引")
            return VectorIndex(dimension=self.dimension)

        logger.info(f"开始构建向量索引: {len(documents)} 个文档")

        # Step 1: 向量化
        logger.info("Step 1/2: 通过 API 文本向量化...")
        embeddings = self.encode(documents, show_progress=show_progress)
        logger.info(f"向量化完成: shape={embeddings.shape}")

        # 自动检测实际维度（以 API 返回为准）
        actual_dim = embeddings.shape[1] if len(embeddings.shape) == 2 else self.dimension
        if actual_dim != self.dimension:
            logger.info(f"实际维度={actual_dim}，更新 dimension 配置")
            self.dimension = actual_dim

        # Step 2: 构建索引
        logger.info("Step 2/2: 构建索引...")
        index = VectorIndex(dimension=self.dimension)
        index.build(embeddings, documents, metadata)
        logger.info(f"索引构建完成: {index.size} 个文档, 维度={self.dimension}")

        return index

    def build_index_from_documents(self, documents: List[Document],
                                    show_progress: bool = True) -> VectorIndex:
        """
        从 Document 对象列表构建向量索引

        自动提取文本和元数据。

        Args:
            documents: Document 对象列表
            show_progress: 是否显示进度

        Returns:
            构建好的 VectorIndex 实例
        """
        texts = [doc.text for doc in documents]
        metadata = [doc.to_dict() for doc in documents]

        return self.build_index(texts, metadata, show_progress)

    # ==========================================
    # 索引持久化
    # ==========================================

    def save_index(self, index: VectorIndex, path: str):
        """
        保存索引到指定路径

        Args:
            index: 要保存的 VectorIndex 实例
            path: 保存目录路径
        """
        index.save(path)
        logger.info(f"向量索引已保存到: {path}")

    def load_index(self, path: str) -> VectorIndex:
        """
        从指定路径加载索引

        Args:
            path: 索引目录路径

        Returns:
            加载的 VectorIndex 实例
        """
        index = VectorIndex.load(path)
        logger.info(f"向量索引已加载: {index.size} 个文档")
        return index

    # ==========================================
    # 检索
    # ==========================================

    def search(self, index: VectorIndex, query: str,
               top_k: int = 5) -> List[SearchResult]:
        """
        在向量索引中检索

        便捷方法：自动编码查询并搜索。

        Args:
            index: 向量索引
            query: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            检索结果列表
        """
        query_vector = self.encode(query)
        return index.search(query_vector, top_k=top_k)

    # ==========================================
    # 批量操作
    # ==========================================

    def build_expert_indices(
        self,
        expert_data: Dict[str, List[Document]],
        base_path: str,
        show_progress: bool = True,
    ) -> Dict[str, VectorIndex]:
        """
        为多个专家分别构建向量索引

        Args:
            expert_data: {expert_id: [Document, ...]} 专家数据字典
            base_path: 保存的根目录
            show_progress: 是否显示进度

        Returns:
            {expert_id: VectorIndex} 索引字典
        """
        indices = {}

        for expert_id, documents in expert_data.items():
            logger.info(
                f"构建专家 '{expert_id}' 的向量索引 "
                f"({len(documents)} 个文档)..."
            )

            # 构建索引
            index = self.build_index_from_documents(documents, show_progress)

            # 保存索引
            expert_path = os.path.join(base_path, expert_id)
            self.save_index(index, expert_path)

            indices[expert_id] = index

        logger.info(f"所有专家索引构建完成: {len(indices)} 个专家")
        return indices

    def __repr__(self) -> str:
        return (
            f"VectorIndexBuilder(model='{self.model}', "
            f"base_url='{self.base_url}')"
        )
