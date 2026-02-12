"""
专家知识库基类 (Base Expert)

定义专家知识库的抽象基类和默认实现。

每个专家知识库包含：
- 向量索引 (VectorIndex): 用于语义检索（神经方法）
- 知识图谱 (KnowledgeGraph): 用于符号推理（符号方法）
- 领域描述 (ExpertDomain): 描述该专家擅长的领域

支持三种检索模式：
- vector: 仅向量检索
- symbolic: 仅图谱查询
- hybrid: 向量 + 图谱 混合检索（默认）
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.models import Document, Entity, Relation
from kb_builder.models import ExpertDomain, KnowledgeGraph, SearchResult, VectorIndex
from utils.config import Config

logger = logging.getLogger(__name__)


class BaseExpert(ABC):
    """
    专家知识库抽象基类

    每个专家负责一个特定的知识领域，提供向量检索和符号推理两种能力。

    子类必须实现：
    - retrieve_vector(): 向量检索
    - retrieve_symbolic(): 符号检索
    """

    def __init__(
        self,
        expert_id: str,
        domain: ExpertDomain,
        kb_path: Optional[str] = None,
    ):
        """
        初始化专家

        Args:
            expert_id: 专家唯一标识符
            domain: 专家领域描述
            kb_path: 知识库文件路径（包含向量索引和图谱）
        """
        self.expert_id = expert_id
        self.domain = domain
        self.kb_path = kb_path

        self.vector_index: Optional[VectorIndex] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None

        # 如果提供了路径，尝试加载
        if kb_path and os.path.exists(kb_path):
            self._load_knowledge_base(kb_path)

    def _load_knowledge_base(self, kb_path: str):
        """
        加载知识库（向量索引 + 知识图谱）

        Args:
            kb_path: 知识库目录路径
        """
        # 加载向量索引
        vector_path = os.path.join(kb_path, "vectors")
        if os.path.exists(vector_path):
            try:
                self.vector_index = VectorIndex.load(vector_path)
                logger.info(
                    f"[{self.expert_id}] 向量索引加载成功: "
                    f"{self.vector_index.size} 个文档"
                )
            except Exception as e:
                logger.warning(f"[{self.expert_id}] 向量索引加载失败: {e}")

        # 加载知识图谱
        graph_path = os.path.join(kb_path, "graph")
        if os.path.exists(graph_path):
            try:
                self.knowledge_graph = KnowledgeGraph.load(graph_path)
                logger.info(
                    f"[{self.expert_id}] 知识图谱加载成功: "
                    f"{self.knowledge_graph.num_entities} 实体, "
                    f"{self.knowledge_graph.num_relations} 关系"
                )
            except Exception as e:
                logger.warning(f"[{self.expert_id}] 知识图谱加载失败: {e}")

    # ==========================================
    # 抽象方法（子类必须实现）
    # ==========================================

    @abstractmethod
    def retrieve_vector(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        向量检索（神经方法）

        使用向量相似度在知识库中检索最相关的文档。

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            检索结果列表
        """
        ...

    @abstractmethod
    def retrieve_symbolic(self, query: Dict[str, Any]) -> List[SearchResult]:
        """
        符号检索（图谱查询）

        在知识图谱上执行结构化查询。

        Args:
            query: 结构化查询字典，支持以下格式：
                - {"entity": "Tesla", "relation": "CEO"}: 单跳查询
                - {"pattern": [("Tesla", "CEO", "?x")]}: 模式匹配
                - {"path": {"start": "A", "end": "B", "max_hops": 3}}: 路径查询

        Returns:
            检索结果列表
        """
        ...

    # ==========================================
    # 混合检索（默认实现）
    # ==========================================

    def hybrid_retrieve(
        self,
        query: str,
        symbolic_query: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        alpha: float = 0.7,
    ) -> List[SearchResult]:
        """
        混合检索：结合向量检索和符号检索

        Args:
            query: 自然语言查询文本
            symbolic_query: 结构化符号查询（可选）
            top_k: 返回前 k 个结果
            alpha: 向量检索权重 (0~1)，1-alpha 为符号检索权重

        Returns:
            融合后的检索结果列表
        """
        results = []

        # 向量检索
        vector_results = []
        if self.vector_index is not None:
            try:
                vector_results = self.retrieve_vector(query, top_k=top_k)
            except Exception as e:
                logger.warning(f"[{self.expert_id}] 向量检索失败: {e}")

        # 符号检索
        symbolic_results = []
        if symbolic_query and self.knowledge_graph is not None:
            try:
                symbolic_results = self.retrieve_symbolic(symbolic_query)
            except Exception as e:
                logger.warning(f"[{self.expert_id}] 符号检索失败: {e}")

        # 融合结果
        if vector_results and symbolic_results:
            results = self._merge_results(
                vector_results, symbolic_results, alpha=alpha
            )
        elif vector_results:
            results = vector_results
        elif symbolic_results:
            results = symbolic_results

        # 标注来源专家
        for result in results:
            result.expert_id = self.expert_id

        return results[:top_k]

    def _merge_results(
        self,
        vector_results: List[SearchResult],
        symbolic_results: List[SearchResult],
        alpha: float = 0.7,
    ) -> List[SearchResult]:
        """
        融合向量检索和符号检索的结果

        策略：
        1. 对两种结果分别归一化分数到 [0, 1]
        2. 加权融合：score = alpha * vector_score + (1 - alpha) * symbolic_score
        3. 相同文档的结果合并，取最高分
        4. 按融合分数排序

        Args:
            vector_results: 向量检索结果
            symbolic_results: 符号检索结果
            alpha: 向量检索权重

        Returns:
            融合排序后的结果列表
        """
        # 归一化向量结果分数
        if vector_results:
            v_max = max(r.score for r in vector_results)
            v_min = min(r.score for r in vector_results)
            v_range = v_max - v_min if v_max > v_min else 1.0
            for r in vector_results:
                r.score = (r.score - v_min) / v_range

        # 归一化符号结果分数
        if symbolic_results:
            s_max = max(r.score for r in symbolic_results)
            s_min = min(r.score for r in symbolic_results)
            s_range = s_max - s_min if s_max > s_min else 1.0
            for r in symbolic_results:
                r.score = (r.score - s_min) / s_range

        # 构建 ID -> 结果 的映射进行合并
        merged: Dict[str, SearchResult] = {}

        for r in vector_results:
            key = r.id or r.text[:80]
            merged[key] = SearchResult(
                id=r.id,
                text=r.text,
                score=alpha * r.score,
                metadata=r.metadata,
                source="hybrid",
                expert_id=self.expert_id,
            )

        for r in symbolic_results:
            key = r.id or r.text[:80]
            if key in merged:
                # 已有向量结果，叠加符号分数
                merged[key].score += (1 - alpha) * r.score
                merged[key].source = "hybrid"
                # 合并元数据
                merged[key].metadata.update(r.metadata)
            else:
                merged[key] = SearchResult(
                    id=r.id,
                    text=r.text,
                    score=(1 - alpha) * r.score,
                    metadata=r.metadata,
                    source="hybrid",
                    expert_id=self.expert_id,
                )

        # 按分数排序
        results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return results

    # ==========================================
    # 属性和工具方法
    # ==========================================

    @property
    def is_loaded(self) -> bool:
        """检查知识库是否已加载"""
        return self.vector_index is not None or self.knowledge_graph is not None

    @property
    def has_vector_index(self) -> bool:
        """是否有向量索引"""
        return self.vector_index is not None and self.vector_index.size > 0

    @property
    def has_knowledge_graph(self) -> bool:
        """是否有知识图谱"""
        return self.knowledge_graph is not None and self.knowledge_graph.num_entities > 0

    def get_statistics(self) -> Dict[str, Any]:
        """获取专家统计信息"""
        stats = {
            "expert_id": self.expert_id,
            "domain": self.domain.to_dict(),
            "vector_index_size": self.vector_index.size if self.vector_index else 0,
            "graph_entities": self.knowledge_graph.num_entities if self.knowledge_graph else 0,
            "graph_relations": self.knowledge_graph.num_relations if self.knowledge_graph else 0,
        }
        return stats

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id='{self.expert_id}', "
            f"domain='{self.domain.name}', loaded={self.is_loaded})"
        )


# ==============================================================================
# 默认专家实现
# ==============================================================================


class DomainExpert(BaseExpert):
    """
    领域专家默认实现

    提供完整的向量检索和符号检索功能。
    可以直接使用，也可以作为更特化专家的基类。

    Usage:
        >>> domain = ExpertDomain(id="expert_0", name="人物领域", ...)
        >>> expert = DomainExpert("expert_0", domain, kb_path="data/knowledge_bases/expert_0")
        >>> results = expert.hybrid_retrieve("Who is the CEO of Tesla?")
    """

    def __init__(
        self,
        expert_id: str,
        domain: ExpertDomain,
        kb_path: Optional[str] = None,
        embedding_builder: Optional[Any] = None,
        config: Optional[Config] = None,
    ):
        """
        初始化领域专家

        Args:
            expert_id: 专家 ID
            domain: 领域描述
            kb_path: 知识库路径
            embedding_builder: VectorIndexBuilder 实例（用于查询编码）
            config: 配置
        """
        super().__init__(expert_id, domain, kb_path)

        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self._config = config
        self._embedding_builder = embedding_builder

    def _get_embedding_builder(self):
        """延迟获取 VectorIndexBuilder（避免循环依赖和提前加载模型）"""
        if self._embedding_builder is None:
            from kb_builder.vector_builder import VectorIndexBuilder
            self._embedding_builder = VectorIndexBuilder(config=self._config)
        return self._embedding_builder

    # ==========================================
    # 向量检索实现
    # ==========================================

    def retrieve_vector(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        向量检索实现

        使用 VectorIndexBuilder 编码查询文本，在向量索引中搜索。

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            检索结果列表
        """
        if not self.has_vector_index:
            logger.debug(f"[{self.expert_id}] 没有向量索引，跳过向量检索")
            return []

        builder = self._get_embedding_builder()
        query_vector = builder.encode(query)

        results = self.vector_index.search(query_vector, top_k=top_k)

        # 标注来源
        for r in results:
            r.source = "vector"
            r.expert_id = self.expert_id

        return results

    # ==========================================
    # 符号检索实现
    # ==========================================

    def retrieve_symbolic(self, query: Dict[str, Any]) -> List[SearchResult]:
        """
        符号检索实现

        支持三种查询模式：
        1. 单跳查询: {"entity": "Tesla", "relation": "CEO"}
        2. 模式匹配: {"pattern": [("Tesla", "CEO", "?person")]}
        3. 路径查询: {"path": {"start": "A", "end": "B", "max_hops": 3}}

        Args:
            query: 结构化查询字典

        Returns:
            检索结果列表
        """
        if not self.has_knowledge_graph:
            logger.debug(f"[{self.expert_id}] 没有知识图谱，跳过符号检索")
            return []

        results = []

        # 模式 1: 单跳查询
        if "entity" in query:
            entity_name = query["entity"]
            relation_type = query.get("relation", None)
            results = self._query_single_hop(entity_name, relation_type)

        # 模式 2: 模式匹配
        elif "pattern" in query:
            pattern = query["pattern"]
            results = self._query_pattern(pattern)

        # 模式 3: 路径查询
        elif "path" in query:
            path_query = query["path"]
            results = self._query_path(
                start=path_query.get("start", ""),
                end=path_query.get("end", ""),
                max_hops=path_query.get("max_hops", 3),
            )

        # 模式 4: 实体搜索
        elif "search" in query:
            search_text = query["search"]
            results = self._search_entity(search_text)

        # 标注来源
        for r in results:
            r.source = "symbolic"
            r.expert_id = self.expert_id

        return results

    def _query_single_hop(self, entity: str,
                           relation_type: Optional[str] = None) -> List[SearchResult]:
        """单跳查询：查找实体的邻居"""
        neighbors = self.knowledge_graph.query_neighbors(
            entity, relation_type=relation_type, direction="out"
        )

        results = []
        for i, neighbor in enumerate(neighbors):
            # 获取具体的关系信息
            relations = self.knowledge_graph.get_relations(
                subject=entity, predicate=relation_type, obj=neighbor.name
            )
            rel_text = relations[0].predicate if relations else "related_to"

            results.append(SearchResult(
                id=neighbor.id,
                text=f"{entity} --[{rel_text}]--> {neighbor.name}",
                score=1.0 - (i * 0.1),  # 递减分数
                metadata={
                    "entity": neighbor.name,
                    "entity_type": neighbor.type,
                    "relation": rel_text,
                    "hop": 1,
                },
            ))

        return results

    def _query_pattern(self, pattern: List[Tuple]) -> List[SearchResult]:
        """模式匹配查询"""
        # 转换为 tuple 列表
        pattern_tuples = [tuple(p) for p in pattern]
        bindings_list = self.knowledge_graph.query_pattern(pattern_tuples)

        results = []
        for i, bindings in enumerate(bindings_list):
            # 构建结果文本
            binding_strs = [f"{k}={v}" for k, v in bindings.items()]
            result_text = ", ".join(binding_strs)

            results.append(SearchResult(
                id=f"pattern_result_{i}",
                text=result_text,
                score=1.0,
                metadata={
                    "bindings": bindings,
                    "pattern": [list(p) for p in pattern_tuples],
                },
            ))

        return results

    def _query_path(self, start: str, end: str,
                     max_hops: int = 3) -> List[SearchResult]:
        """路径查询：查找两个实体间的路径"""
        paths = self.knowledge_graph.query_path(start, end, max_hops=max_hops)

        results = []
        for i, path in enumerate(paths):
            # 构建路径文本
            path_strs = [f"{s} --[{p}]--> {o}" for s, p, o in path]
            path_text = " | ".join(path_strs)

            results.append(SearchResult(
                id=f"path_result_{i}",
                text=path_text,
                score=1.0 / len(path),  # 路径越短分数越高
                metadata={
                    "path": [list(triple) for triple in path],
                    "hops": len(path),
                    "start": start,
                    "end": end,
                },
            ))

        # 按路径长度排序（短路径优先）
        results.sort(key=lambda r: r.metadata.get("hops", 999))
        return results

    def _search_entity(self, text: str, top_k: int = 5) -> List[SearchResult]:
        """实体名称模糊搜索"""
        query_lower = text.lower().strip()
        results = []

        for name, data in self.knowledge_graph.graph.nodes(data=True):
            name_lower = name.lower()
            score = 0.0

            if name_lower == query_lower:
                score = 1.0
            elif query_lower in name_lower:
                score = len(query_lower) / len(name_lower)
            elif name_lower in query_lower:
                score = len(name_lower) / len(query_lower) * 0.8
            else:
                continue

            results.append(SearchResult(
                id=data.get("id", name),
                text=name,
                score=score,
                metadata={
                    "type": data.get("type", "unknown"),
                    "attributes": data.get("attributes", {}),
                },
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ==========================================
    # 知识库设置（用于构建阶段）
    # ==========================================

    def set_vector_index(self, index: VectorIndex):
        """设置向量索引"""
        self.vector_index = index
        logger.info(f"[{self.expert_id}] 向量索引已设置: {index.size} 个文档")

    def set_knowledge_graph(self, graph: KnowledgeGraph):
        """设置知识图谱"""
        self.knowledge_graph = graph
        logger.info(
            f"[{self.expert_id}] 知识图谱已设置: "
            f"{graph.num_entities} 实体, {graph.num_relations} 关系"
        )

    def save(self, path: Optional[str] = None):
        """
        保存专家知识库

        保存结构：
        - {path}/vectors/: 向量索引
        - {path}/graph/: 知识图谱
        - {path}/metadata.json: 专家元数据

        Args:
            path: 保存目录路径，None 则使用 self.kb_path
        """
        save_path = path or self.kb_path
        if not save_path:
            raise ValueError("未指定保存路径")

        os.makedirs(save_path, exist_ok=True)

        # 保存向量索引
        if self.vector_index is not None:
            vector_path = os.path.join(save_path, "vectors")
            self.vector_index.save(vector_path)

        # 保存知识图谱
        if self.knowledge_graph is not None:
            graph_path = os.path.join(save_path, "graph")
            self.knowledge_graph.save(graph_path)

        # 保存元数据
        metadata = {
            "expert_id": self.expert_id,
            "domain": self.domain.to_dict(),
            "vector_index_size": self.vector_index.size if self.vector_index else 0,
            "graph_entities": self.knowledge_graph.num_entities if self.knowledge_graph else 0,
            "graph_relations": self.knowledge_graph.num_relations if self.knowledge_graph else 0,
        }
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.kb_path = save_path
        logger.info(f"[{self.expert_id}] 专家知识库已保存到: {save_path}")

    @classmethod
    def load_from_path(cls, path: str, config: Optional[Config] = None) -> DomainExpert:
        """
        从路径加载专家

        Args:
            path: 知识库目录路径
            config: 配置对象

        Returns:
            加载好的 DomainExpert 实例
        """
        # 加载元数据
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"专家元数据文件不存在: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        expert_id = metadata.get("expert_id", os.path.basename(path))
        domain = ExpertDomain.from_dict(metadata.get("domain", {}))

        expert = cls(
            expert_id=expert_id,
            domain=domain,
            kb_path=path,
            config=config,
        )

        logger.info(
            f"专家 '{expert_id}' 已从 {path} 加载 "
            f"(向量={expert.has_vector_index}, 图谱={expert.has_knowledge_graph})"
        )
        return expert
