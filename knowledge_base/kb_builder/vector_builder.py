"""
向量索引构建器 (Vector Index Builder)

负责将文本文档转换为向量表示，并构建高效的向量检索索引。

核心流程：
1. 通过 OpenAI 兼容 API 调用 Embedding 模型
2. 文本逐条向量化（每次请求处理一条文本）
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
    - 每次处理一条文本（避免批量调用问题）
    - 自动重试机制
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

    def _generate_curl_command(self, url: str, headers: Dict[str, str], body: Dict) -> str:
        """
        生成等效的 curl 命令用于调试
        
        Args:
            url: 请求 URL
            headers: 请求头
            body: 请求体
            
        Returns:
            curl 命令字符串
        """
        import json
        
        # 构造 curl 命令
        curl_parts = ["curl -X POST"]
        
        # 添加 URL
        curl_parts.append(f'"{url}"')
        
        # 添加请求头
        for key, value in headers.items():
            # 对于 Authorization,隐藏部分 key
            if key == "Authorization" and "Bearer" in value:
                api_key = value.replace("Bearer ", "")
                if len(api_key) > 30:
                    masked_key = f"{api_key[:20]}...{api_key[-10:]}"
                else:
                    masked_key = api_key
                curl_parts.append(f'-H "{key}: Bearer {masked_key}"')
            else:
                curl_parts.append(f'-H "{key}: {value}"')
        
        # 添加请求体 (只包含前2条数据用于测试)
        test_body = body.copy()
        if "input" in test_body and isinstance(test_body["input"], list):
            # 只取前2条,并且限制每条文本长度
            original_inputs = test_body["input"]
            test_body["input"] = [
                text[:100] + "..." if len(text) > 100 else text 
                for text in original_inputs[:2]
            ]
        
        # 生成 JSON,确保中文正常显示
        body_json = json.dumps(test_body, ensure_ascii=False)
        
        # 为不同操作系统生成不同的转义格式
        # Linux/Mac: 单引号包裹,内部双引号不需要转义
        # Windows PowerShell: 需要特殊处理
        import platform
        if platform.system() == "Windows":
            # Windows PowerShell 版本
            body_json_escaped = body_json.replace('"', '""')
            curl_parts.append(f"-d '{body_json}'")
            
            # 额外生成一个 PowerShell 版本的说明
            ps_body = body_json.replace("'", "''")
            ps_cmd = f"""
  # PowerShell 版本:
  $body = '{ps_body}'
  Invoke-RestMethod -Uri "{url}" -Method Post -Body $body -ContentType "application/json" """
            curl_command = " \\\n  ".join(curl_parts)
            return curl_command + "\n" + ps_cmd
        else:
            # Linux/Mac 版本
            body_json_escaped = body_json.replace("'", "'\\''")
            curl_parts.append(f"-d '{body_json_escaped}'")
            return " \\\n  ".join(curl_parts)

    # ==========================================
    # 向量编码
    # ==========================================

    def encode(self, texts: str | List[str],
               show_progress: bool = False) -> np.ndarray:
        """
        将文本编码为向量（通过 API 调用）
        每次只处理一条文本

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

        # 逐条调用 API
        all_embeddings: List[Optional[List[float]]] = []
        detected_dim: Optional[int] = None  # 从成功响应中检测的实际维度
        failed_indices: List[int] = []  # all_embeddings 中需要回填的下标

        iterator = texts
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Embedding API 调用")
            except ImportError:
                pass

        client = self._get_client()

        for idx, text in enumerate(iterator):
            # 清理文本：去除空字符串，截断过长文本
            cleaned_text = text.strip()[:8000] if text.strip() else " "

            try:
                # 添加请求详情日志（仅首次）
                if idx == 0:
                    logger.info(
                        f"首次 Embedding 请求: "
                        f"text_len={len(cleaned_text)}, "
                        f"model={self.model}"
                    )
                
                response = client.embeddings.create(
                    input=[cleaned_text],  # 每次只传入一条文本
                    model=self.model,
                )

                embedding = response.data[0].embedding

                # 从第一次成功响应中检测实际维度
                if detected_dim is None:
                    detected_dim = len(embedding)
                    if detected_dim != self.dimension:
                        logger.info(
                            f"API 返回的实际维度={detected_dim}，"
                            f"与配置维度={self.dimension} 不同，以 API 为准"
                        )

                all_embeddings.append(embedding)

            except Exception as e:
                # 详细的错误信息
                import json
                
                error_details = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "api_config": {
                        "base_url": self.base_url,
                        "model": self.model,
                    },
                    "text_info": {
                        "text_preview": cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text,
                        "text_len": len(cleaned_text),
                    }
                }
                
                # 构造完整的请求信息用于调试
                request_url = f"{self.base_url.rstrip('/')}/embeddings"
                request_headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                # 请求体
                display_text = cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text
                request_body = {
                    "input": [display_text],
                    "model": self.model
                }
                
                # 格式化 JSON
                request_body_str = json.dumps(request_body, ensure_ascii=False, indent=2)
                
                # 生成等效的 curl 命令
                full_request_body = {
                    "input": [cleaned_text],
                    "model": self.model
                }
                curl_command = self._generate_curl_command(
                    url=request_url,
                    headers=request_headers,
                    body=full_request_body
                )
                
                # 输出详细错误信息
                logger.error(
                    f"Embedding API 调用失败 (文本索引 {idx}):\n"
                    f"  错误类型: {error_details['error_type']}\n"
                    f"  错误信息: {error_details['error_message']}\n"
                    f"  API 地址: {error_details['api_config']['base_url']}\n"
                    f"  模型名称: {error_details['api_config']['model']}\n"
                    f"  文本预览: {error_details['text_info']['text_preview']}\n"
                    f"\n"
                    f"  === 完整请求信息 ===\n"
                    f"  URL: {request_url}\n"
                    f"  Headers:\n"
                    f"    Content-Type: application/json\n"
                    f"    Authorization: Bearer {self.api_key[:15]}...{self.api_key[-8:] if len(self.api_key) > 25 else ''}\n"
                    f"  Body:\n"
                    f"{request_body_str}\n"
                    f"\n"
                    f"  === 等效调试命令 ===\n"
                    f"{curl_command}\n"
                )
                
                # 记录失败位置，用 None 占位
                failed_indices.append(idx)
                all_embeddings.append(None)

        # 确定最终维度：优先使用 API 返回的实际维度
        final_dim = detected_dim if detected_dim is not None else self.dimension

        # 回填失败的占位向量
        if failed_indices:
            logger.warning(
                f"共 {len(failed_indices)} 条文本 Embedding 失败，"
                f"使用维度={final_dim} 的零向量填充"
            )
            zero_vec = [0.0] * final_dim
            for idx in failed_indices:
                all_embeddings[idx] = zero_vec

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
