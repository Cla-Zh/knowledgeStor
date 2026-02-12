"""
专家路由器 (Expert Router)

决定用户查询应该路由到哪个/哪些专家知识库。

支持的路由策略：
1. similarity: 基于查询与专家领域描述的语义相似度（默认）
2. keyword: 基于关键词匹配
3. hybrid: 相似度 + 关键词混合策略

特别支持多跳查询路由：为推理链的每一跳分别选择最合适的专家。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experts.base_expert import BaseExpert
from kb_builder.models import ExpertDomain
from utils.config import Config

logger = logging.getLogger(__name__)


class ExpertRouter:
    """
    专家路由器

    根据查询内容，将请求路由到最相关的专家知识库。

    Usage:
        >>> experts = [expert_0, expert_1, expert_2]
        >>> router = ExpertRouter(experts)
        >>> selected = router.route("Who is the CEO of Tesla?", top_k_experts=2)
        >>> # selected = [expert_2 (组织), expert_0 (人物)]
    """

    def __init__(
        self,
        experts: List[BaseExpert],
        routing_model: str = "similarity",
        config: Optional[Config] = None,
    ):
        """
        初始化专家路由器

        Args:
            experts: 可用的专家列表
            routing_model: 路由策略 - 'similarity', 'keyword', 'hybrid'
            config: 配置对象
        """
        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self.experts = experts
        self.routing_model = routing_model
        self._config = config

        # 用于相似度路由的编码器（延迟加载）
        self._encoder = None

        # 预计算领域描述的向量（延迟初始化）
        self._domain_embeddings: Optional[np.ndarray] = None
        self._domain_texts: List[str] = []

        logger.info(
            f"ExpertRouter 初始化: {len(experts)} 个专家, "
            f"策略={routing_model}"
        )

    # ==========================================
    # 主路由接口
    # ==========================================

    def route(self, query: str, top_k_experts: int = 2) -> List[BaseExpert]:
        """
        将查询路由到最相关的专家

        Args:
            query: 用户查询文本
            top_k_experts: 返回前 k 个最相关的专家

        Returns:
            按相关度排序的专家列表
        """
        if not self.experts:
            logger.warning("没有可用的专家")
            return []

        if len(self.experts) <= top_k_experts:
            return list(self.experts)

        # 计算各专家的相关度分数
        scores = self._compute_routing_scores(query)

        # 按分数排序，取 top_k
        scored_experts = sorted(
            zip(self.experts, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        selected = [expert for expert, score in scored_experts[:top_k_experts]]

        logger.info(
            f"路由结果: query='{query[:50]}...' -> "
            f"{[e.expert_id for e in selected]} "
            f"(scores={[f'{s:.3f}' for _, s in scored_experts[:top_k_experts]]})"
        )

        return selected

    def route_with_scores(self, query: str,
                          top_k_experts: int = 2) -> List[Tuple[BaseExpert, float]]:
        """
        路由并返回各专家的得分

        Args:
            query: 查询文本
            top_k_experts: 返回前 k 个

        Returns:
            [(expert, score), ...] 列表
        """
        if not self.experts:
            return []

        scores = self._compute_routing_scores(query)
        scored_experts = sorted(
            zip(self.experts, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return scored_experts[:top_k_experts]

    def route_multi_hop(
        self,
        query: str,
        reasoning_chain: List[str],
        top_k_experts: int = 2,
    ) -> List[List[BaseExpert]]:
        """
        多跳查询路由

        为推理链的每一跳分别选择最合适的专家。

        Args:
            query: 原始查询
            reasoning_chain: 推理链中每一跳的子查询文本
            top_k_experts: 每跳选择的专家数量

        Returns:
            [[hop1_experts], [hop2_experts], ...] 每跳的专家列表

        Example:
            >>> chain = ["Who is the CEO of Tesla?", "Where did Elon Musk study?"]
            >>> hop_experts = router.route_multi_hop(query, chain)
            >>> # hop_experts[0] = [org_expert, person_expert]
            >>> # hop_experts[1] = [person_expert, org_expert]
        """
        if not reasoning_chain:
            # 没有推理链，对原始查询做单次路由
            return [self.route(query, top_k_experts)]

        hop_experts = []
        for hop_idx, sub_query in enumerate(reasoning_chain):
            experts = self.route(sub_query, top_k_experts)
            hop_experts.append(experts)
            logger.debug(
                f"Hop {hop_idx + 1}: '{sub_query[:40]}...' -> "
                f"{[e.expert_id for e in experts]}"
            )

        return hop_experts

    # ==========================================
    # 路由策略实现
    # ==========================================

    def _compute_routing_scores(self, query: str) -> List[float]:
        """
        计算查询与各专家的路由分数

        根据 routing_model 选择不同策略。

        Args:
            query: 查询文本

        Returns:
            每个专家的分数列表（与 self.experts 对齐）
        """
        if self.routing_model == "similarity":
            return self._score_by_similarity(query)
        elif self.routing_model == "keyword":
            return self._score_by_keyword(query)
        elif self.routing_model == "hybrid":
            sim_scores = self._score_by_similarity(query)
            kw_scores = self._score_by_keyword(query)
            # 混合：0.6 * 语义相似度 + 0.4 * 关键词匹配
            return [
                0.6 * s + 0.4 * k
                for s, k in zip(sim_scores, kw_scores)
            ]
        else:
            logger.warning(f"未知路由策略 '{self.routing_model}'，回退到 keyword")
            return self._score_by_keyword(query)

    # ----- 策略 1: 语义相似度 -----

    def _score_by_similarity(self, query: str) -> List[float]:
        """
        基于语义相似度的路由评分

        使用 Embedding 模型编码查询和领域描述，计算余弦相似度。

        Args:
            query: 查询文本

        Returns:
            分数列表
        """
        # 确保领域向量已初始化
        self._ensure_domain_embeddings()

        if self._domain_embeddings is None or len(self._domain_embeddings) == 0:
            # 回退到关键词方法
            return self._score_by_keyword(query)

        # 编码查询
        encoder = self._get_encoder()
        query_embedding = encoder.encode(query)  # shape: (1, dim) or (dim,)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # 计算余弦相似度
        scores = self._cosine_similarity(query_embedding, self._domain_embeddings)
        return scores.flatten().tolist()

    def _ensure_domain_embeddings(self):
        """确保领域描述向量已计算"""
        if self._domain_embeddings is not None:
            return

        if not self.experts:
            return

        # 检查是否有预计算的 centroid
        centroids = []
        for expert in self.experts:
            if expert.domain.centroid is not None:
                centroids.append(expert.domain.centroid)

        if len(centroids) == len(self.experts):
            # 所有专家都有 centroid，直接使用
            self._domain_embeddings = np.vstack(centroids).astype(np.float32)
            logger.info("使用预计算的领域中心向量进行路由")
            return

        # 否则用领域描述文本编码
        try:
            self._domain_texts = []
            for expert in self.experts:
                # 构建领域描述文本
                domain = expert.domain
                desc_parts = [
                    domain.name,
                    domain.description,
                    " ".join(domain.entity_types),
                    " ".join(domain.keywords),
                ]
                self._domain_texts.append(" ".join(desc_parts))

            encoder = self._get_encoder()
            self._domain_embeddings = encoder.encode(
                self._domain_texts
            ).astype(np.float32)
            logger.info(
                f"已编码 {len(self._domain_texts)} 个领域描述, "
                f"shape={self._domain_embeddings.shape}"
            )
        except Exception as e:
            logger.warning(f"领域描述编码失败: {e}，路由将回退到关键词方法")
            self._domain_embeddings = None

    def _get_encoder(self):
        """获取 Embedding 编码器"""
        if self._encoder is None:
            from kb_builder.vector_builder import VectorIndexBuilder
            self._encoder = VectorIndexBuilder(config=self._config)
        return self._encoder

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T)

    # ----- 策略 2: 关键词匹配 -----

    def _score_by_keyword(self, query: str) -> List[float]:
        """
        基于关键词匹配的路由评分

        计算查询文本中包含的领域关键词数量。

        Args:
            query: 查询文本

        Returns:
            分数列表
        """
        query_lower = query.lower()
        scores = []

        for expert in self.experts:
            domain = expert.domain
            score = 0.0

            # 关键词匹配（权重 0.5）
            if domain.keywords:
                keyword_hits = sum(
                    1 for kw in domain.keywords if kw.lower() in query_lower
                )
                score += 0.5 * (keyword_hits / len(domain.keywords))

            # 实体类型匹配（权重 0.3）
            if domain.entity_types:
                type_hits = sum(
                    1 for et in domain.entity_types if et.lower() in query_lower
                )
                score += 0.3 * (type_hits / len(domain.entity_types))

            # 领域名称匹配（权重 0.2）
            if domain.name and domain.name.lower() in query_lower:
                score += 0.2

            scores.append(score)

        # 避免全零
        if all(s == 0 for s in scores):
            scores = [1.0 / len(self.experts)] * len(self.experts)

        return scores

    # ==========================================
    # 专家管理
    # ==========================================

    def add_expert(self, expert: BaseExpert):
        """
        添加新专家

        Args:
            expert: 专家实例
        """
        self.experts.append(expert)
        # 重置缓存的领域向量
        self._domain_embeddings = None
        self._domain_texts = []
        logger.info(f"添加专家: {expert.expert_id}")

    def remove_expert(self, expert_id: str) -> bool:
        """
        移除专家

        Args:
            expert_id: 要移除的专家 ID

        Returns:
            是否成功移除
        """
        for i, expert in enumerate(self.experts):
            if expert.expert_id == expert_id:
                self.experts.pop(i)
                self._domain_embeddings = None
                self._domain_texts = []
                logger.info(f"移除专家: {expert_id}")
                return True
        return False

    def get_expert(self, expert_id: str) -> Optional[BaseExpert]:
        """根据 ID 获取专家"""
        for expert in self.experts:
            if expert.expert_id == expert_id:
                return expert
        return None

    def get_all_statistics(self) -> List[Dict[str, Any]]:
        """获取所有专家的统计信息"""
        return [expert.get_statistics() for expert in self.experts]

    def __repr__(self) -> str:
        expert_ids = [e.expert_id for e in self.experts]
        return (
            f"ExpertRouter(experts={expert_ids}, "
            f"strategy='{self.routing_model}')"
        )
