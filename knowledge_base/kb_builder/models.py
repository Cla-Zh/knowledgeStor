"""
知识库构建模型 (KB Builder Models)

定义知识库构建和检索相关的数据结构，包括：
- ExpertDomain: 专家领域描述
- SearchResult: 检索结果
- VectorIndex: 向量索引封装（基于 FAISS）
- KnowledgeGraph: 知识图谱封装（基于 NetworkX）
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from data.models import Entity, Relation

logger = logging.getLogger(__name__)


# ==============================================================================
# 专家领域模型
# ==============================================================================


@dataclass
class ExpertDomain:
    """
    专家领域描述

    描述一个专家知识库所擅长的领域信息。

    Attributes:
        id: 专家领域唯一标识符
        name: 领域名称（如 "人物领域"、"地理领域"）
        description: 领域描述
        entity_types: 该专家擅长的实体类型列表
        keywords: 领域关键词列表
        data_count: 该领域的数据量
        centroid: 领域向量中心（用于路由时的相似度计算）
    """

    id: str = ""
    name: str = ""
    description: str = ""
    entity_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    data_count: int = 0
    centroid: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（不含 centroid 向量）"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "entity_types": self.entity_types,
            "keywords": self.keywords,
            "data_count": self.data_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExpertDomain:
        """从字典反序列化"""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            entity_types=data.get("entity_types", []),
            keywords=data.get("keywords", []),
            data_count=data.get("data_count", 0),
        )

    def __repr__(self) -> str:
        return (
            f"ExpertDomain(id='{self.id}', name='{self.name}', "
            f"types={self.entity_types}, data_count={self.data_count})"
        )


# ==============================================================================
# 检索结果模型
# ==============================================================================


@dataclass
class SearchResult:
    """
    检索结果

    Attributes:
        id: 结果对应的文档/实体 ID
        text: 结果文本内容
        score: 相似度/相关性分数
        metadata: 附加元数据
        source: 结果来源（如 "vector", "graph", "hybrid"）
        expert_id: 来源专家 ID
    """

    id: str = ""
    text: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    expert_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
            "expert_id": self.expert_id,
        }

    def __repr__(self) -> str:
        text_preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        return (
            f"SearchResult(score={self.score:.4f}, source='{self.source}', "
            f"text='{text_preview}')"
        )


# ==============================================================================
# 向量索引封装
# ==============================================================================


class VectorIndex:
    """
    向量索引封装

    基于 FAISS 的向量索引，支持高效的最近邻搜索。
    同时维护元数据映射，将向量 ID 关联到文档信息。

    Attributes:
        index: FAISS 索引实例
        embeddings: 所有文档的向量矩阵
        documents: 文档文本列表
        metadata: 每个文档的元数据列表
        dimension: 向量维度
    """

    def __init__(self, dimension: int = 384):
        """
        初始化向量索引

        Args:
            dimension: 向量维度（默认 384，对应 all-MiniLM-L6-v2）
        """
        self.index = None               # FAISS Index 实例
        self.embeddings: np.ndarray = np.array([])  # 向量矩阵
        self.documents: List[str] = []   # 文档文本
        self.metadata: List[Dict] = []   # 元数据
        self.dimension: int = dimension
        self._id_map: Dict[int, str] = {}  # 内部索引 -> 文档 ID 映射

    @property
    def size(self) -> int:
        """索引中的文档数量"""
        if self.index is not None:
            return self.index.ntotal
        return len(self.documents)

    def build(self, embeddings: np.ndarray, documents: List[str],
              metadata: Optional[List[Dict]] = None):
        """
        构建向量索引

        Args:
            embeddings: 向量矩阵，shape=(n_docs, dimension)
            documents: 文档文本列表
            metadata: 元数据列表（可选）

        Raises:
            ValueError: 如果输入数据不一致
        """
        try:
            import faiss
        except ImportError:
            logger.warning(
                "FAISS 未安装，将使用基于 numpy 的简单索引。"
                "建议安装: pip install faiss-cpu"
            )
            faiss = None

        if len(embeddings) != len(documents):
            raise ValueError(
                f"向量数 ({len(embeddings)}) 与文档数 ({len(documents)}) 不匹配"
            )

        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]
        self.dimension = self.embeddings.shape[1] if len(self.embeddings) > 0 else self.dimension

        # 构建 FAISS 索引
        if faiss is not None and len(self.embeddings) > 0:
            # 使用 L2 距离的平面索引（精确搜索）
            self.index = faiss.IndexFlatIP(self.dimension)
            # 归一化向量以使用余弦相似度
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            logger.info(
                f"FAISS 索引构建完成: {self.index.ntotal} 个向量, "
                f"维度={self.dimension}"
            )
        else:
            self.index = None
            logger.info(
                f"使用 numpy 简单索引: {len(self.embeddings)} 个向量, "
                f"维度={self.dimension}"
            )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        搜索最相似的文档

        Args:
            query_vector: 查询向量，shape=(dimension,) 或 (1, dimension)
            top_k: 返回前 k 个结果

        Returns:
            按相似度排序的检索结果列表
        """
        if len(self.documents) == 0:
            return []

        query_vec = np.array(query_vector, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        top_k = min(top_k, len(self.documents))

        try:
            import faiss
            has_faiss = True
        except ImportError:
            has_faiss = False

        if self.index is not None and has_faiss:
            # 使用 FAISS 搜索
            faiss.normalize_L2(query_vec)
            scores, indices = self.index.search(query_vec, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # 使用 numpy 余弦相似度
            scores = self._cosine_similarity(query_vec, self.embeddings)
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append(SearchResult(
                id=self.metadata[idx].get("id", str(idx)),
                text=self.documents[idx],
                score=float(score),
                metadata=self.metadata[idx],
                source="vector",
            ))

        return results

    def _cosine_similarity(self, query: np.ndarray,
                           vectors: np.ndarray) -> np.ndarray:
        """
        计算余弦相似度

        Args:
            query: 查询向量，shape=(1, dim)
            vectors: 文档向量矩阵，shape=(n, dim)

        Returns:
            相似度数组
        """
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(query_norm, vectors_norm.T).flatten()
        return similarities

    def save(self, path: str):
        """
        保存向量索引到文件

        保存结构：
        - {path}/faiss_index.bin: FAISS 索引
        - {path}/vector_data.pkl: 文档、元数据和其他信息

        Args:
            path: 保存目录路径
        """
        os.makedirs(path, exist_ok=True)

        # 保存 FAISS 索引
        if self.index is not None:
            try:
                import faiss
                index_path = os.path.join(path, "faiss_index.bin")
                faiss.write_index(self.index, index_path)
                logger.info(f"FAISS 索引已保存到: {index_path}")
            except ImportError:
                logger.warning("FAISS 未安装，无法保存 FAISS 索引")

        # 保存其他数据
        data = {
            "embeddings": self.embeddings,
            "documents": self.documents,
            "metadata": self.metadata,
            "dimension": self.dimension,
        }
        data_path = os.path.join(path, "vector_data.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"向量数据已保存到: {data_path}")

    @classmethod
    def load(cls, path: str) -> VectorIndex:
        """
        从文件加载向量索引

        Args:
            path: 保存目录路径

        Returns:
            加载的 VectorIndex 实例
        """
        # 加载向量数据
        data_path = os.path.join(path, "vector_data.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        instance = cls(dimension=data.get("dimension", 384))
        instance.embeddings = data.get("embeddings", np.array([]))
        instance.documents = data.get("documents", [])
        instance.metadata = data.get("metadata", [])

        # 尝试加载 FAISS 索引
        index_path = os.path.join(path, "faiss_index.bin")
        if os.path.exists(index_path):
            try:
                import faiss
                instance.index = faiss.read_index(index_path)
                logger.info(
                    f"FAISS 索引已加载: {instance.index.ntotal} 个向量"
                )
            except ImportError:
                logger.warning("FAISS 未安装，将使用 numpy 简单索引")
                instance.index = None
        else:
            instance.index = None

        return instance

    def __repr__(self) -> str:
        return (
            f"VectorIndex(size={self.size}, dimension={self.dimension})"
        )


# ==============================================================================
# 知识图谱封装
# ==============================================================================


class KnowledgeGraph:
    """
    知识图谱封装（基于 NetworkX）

    使用有向多重图（MultiDiGraph）存储知识图谱：
    - 节点：实体（Entity），存储实体属性
    - 边：关系（Relation），支持同两个实体间的多条关系

    支持的操作：
    - 实体/关系的增删查改
    - 多跳路径查询
    - 邻居查询
    - 子图提取
    - 图的序列化和反序列化
    """

    def __init__(self):
        """初始化空的知识图谱"""
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entity_index: Dict[str, Entity] = {}  # name -> Entity 快速索引

    @property
    def num_entities(self) -> int:
        """实体（节点）数量"""
        return self.graph.number_of_nodes()

    @property
    def num_relations(self) -> int:
        """关系（边）数量"""
        return self.graph.number_of_edges()

    # ----- 实体操作 -----

    def add_entity(self, entity: Entity):
        """
        添加实体节点

        Args:
            entity: 实体对象
        """
        self.graph.add_node(
            entity.name,
            id=entity.id,
            type=entity.type,
            attributes=entity.attributes,
            aliases=entity.aliases,
            source=entity.source,
        )
        self._entity_index[entity.name] = entity
        # 为别名也建立索引
        for alias in entity.aliases:
            self._entity_index[alias] = entity

    def get_entity(self, name: str) -> Optional[Entity]:
        """
        根据名称获取实体

        Args:
            name: 实体名称（或别名）

        Returns:
            实体对象，未找到返回 None
        """
        # 先查快速索引
        if name in self._entity_index:
            return self._entity_index[name]
        # 再检查图中是否有该节点
        if name in self.graph:
            node_data = self.graph.nodes[name]
            return Entity(
                id=node_data.get("id", ""),
                name=name,
                type=node_data.get("type", "other"),
                attributes=node_data.get("attributes", {}),
                aliases=node_data.get("aliases", []),
                source=node_data.get("source", ""),
            )
        return None

    def has_entity(self, name: str) -> bool:
        """判断实体是否存在"""
        return name in self.graph or name in self._entity_index

    def get_all_entities(self) -> List[Entity]:
        """获取所有实体"""
        entities = []
        for name, data in self.graph.nodes(data=True):
            entities.append(Entity(
                id=data.get("id", ""),
                name=name,
                type=data.get("type", "other"),
                attributes=data.get("attributes", {}),
                aliases=data.get("aliases", []),
                source=data.get("source", ""),
            ))
        return entities

    # ----- 关系操作 -----

    def add_relation(self, relation: Relation):
        """
        添加关系边

        如果主语或宾语实体不存在于图中，会自动创建节点。

        Args:
            relation: 关系对象
        """
        # 确保节点存在
        if not self.graph.has_node(relation.subject):
            self.graph.add_node(relation.subject, type="unknown", attributes={})
        if not self.graph.has_node(relation.object):
            self.graph.add_node(relation.object, type="unknown", attributes={})

        # 添加有向边
        self.graph.add_edge(
            relation.subject,
            relation.object,
            predicate=relation.predicate,
            confidence=relation.confidence,
            source=relation.source,
            attributes=relation.attributes,
        )

    def get_relations(self, subject: str = None, predicate: str = None,
                      obj: str = None) -> List[Relation]:
        """
        查询关系

        支持按主语、谓词、宾语中的任意组合过滤。

        Args:
            subject: 主语实体名称（可选）
            predicate: 关系谓词（可选）
            obj: 宾语实体名称（可选）

        Returns:
            匹配的关系列表
        """
        results = []

        if subject and obj:
            # 查询两个特定实体间的关系
            if self.graph.has_node(subject) and self.graph.has_node(obj):
                edges = self.graph.get_edge_data(subject, obj)
                if edges:
                    for key, data in edges.items():
                        if predicate and data.get("predicate") != predicate:
                            continue
                        results.append(Relation(
                            subject=subject,
                            predicate=data.get("predicate", ""),
                            object=obj,
                            confidence=data.get("confidence", 1.0),
                            source=data.get("source", ""),
                            attributes=data.get("attributes", {}),
                        ))
        elif subject:
            # 查询特定主语的所有关系
            if self.graph.has_node(subject):
                for _, target, data in self.graph.out_edges(subject, data=True):
                    if predicate and data.get("predicate") != predicate:
                        continue
                    results.append(Relation(
                        subject=subject,
                        predicate=data.get("predicate", ""),
                        object=target,
                        confidence=data.get("confidence", 1.0),
                        source=data.get("source", ""),
                        attributes=data.get("attributes", {}),
                    ))
        elif obj:
            # 查询特定宾语的所有关系
            if self.graph.has_node(obj):
                for source_node, _, data in self.graph.in_edges(obj, data=True):
                    if predicate and data.get("predicate") != predicate:
                        continue
                    results.append(Relation(
                        subject=source_node,
                        predicate=data.get("predicate", ""),
                        object=obj,
                        confidence=data.get("confidence", 1.0),
                        source=data.get("source", ""),
                        attributes=data.get("attributes", {}),
                    ))
        else:
            # 查询所有关系
            for u, v, data in self.graph.edges(data=True):
                if predicate and data.get("predicate") != predicate:
                    continue
                results.append(Relation(
                    subject=u,
                    predicate=data.get("predicate", ""),
                    object=v,
                    confidence=data.get("confidence", 1.0),
                    source=data.get("source", ""),
                    attributes=data.get("attributes", {}),
                ))

        return results

    # ----- 图查询操作 -----

    def query_path(self, start: str, end: str,
                   max_hops: int = 3) -> List[List[Tuple[str, str, str]]]:
        """
        查找两个实体之间的所有路径

        Args:
            start: 起始实体名称
            end: 目标实体名称
            max_hops: 最大跳数限制

        Returns:
            路径列表，每条路径为 [(subject, predicate, object), ...] 的三元组列表
        """
        if not self.graph.has_node(start) or not self.graph.has_node(end):
            return []

        paths = []
        try:
            # 使用 NetworkX 查找所有简单路径
            for path_nodes in nx.all_simple_paths(
                self.graph, start, end, cutoff=max_hops
            ):
                path_triples = []
                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i + 1]
                    edge_data = self.graph.get_edge_data(u, v)
                    if edge_data:
                        # 取第一条边的谓词
                        first_edge = list(edge_data.values())[0]
                        predicate = first_edge.get("predicate", "related_to")
                    else:
                        predicate = "related_to"
                    path_triples.append((u, predicate, v))
                paths.append(path_triples)
        except nx.NetworkXError:
            pass

        return paths

    def query_neighbors(self, entity: str,
                        relation_type: Optional[str] = None,
                        direction: str = "out") -> List[Entity]:
        """
        查询实体的邻居

        Args:
            entity: 实体名称
            relation_type: 关系类型过滤（可选）
            direction: 方向 - "out"（出边）, "in"（入边）, "both"（双向）

        Returns:
            邻居实体列表
        """
        if not self.graph.has_node(entity):
            return []

        neighbors = set()

        if direction in ("out", "both"):
            for _, target, data in self.graph.out_edges(entity, data=True):
                if relation_type and data.get("predicate") != relation_type:
                    continue
                neighbors.add(target)

        if direction in ("in", "both"):
            for source_node, _, data in self.graph.in_edges(entity, data=True):
                if relation_type and data.get("predicate") != relation_type:
                    continue
                neighbors.add(source_node)

        # 将名称转换为 Entity 对象
        result = []
        for name in neighbors:
            ent = self.get_entity(name)
            if ent:
                result.append(ent)
            else:
                result.append(Entity(name=name, type="unknown"))

        return result

    def subgraph(self, entities: List[str]) -> KnowledgeGraph:
        """
        提取包含指定实体的子图

        Args:
            entities: 实体名称列表

        Returns:
            新的 KnowledgeGraph 子图实例
        """
        # 过滤出存在于图中的实体
        valid_entities = [e for e in entities if self.graph.has_node(e)]
        sub = KnowledgeGraph()
        sub.graph = self.graph.subgraph(valid_entities).copy()

        # 重建实体索引
        for name in sub.graph.nodes():
            if name in self._entity_index:
                sub._entity_index[name] = self._entity_index[name]

        return sub

    def query_pattern(self, pattern: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        """
        模式匹配查询（类似 SPARQL）

        支持变量（以 "?" 开头），在知识图谱中查找匹配的绑定。

        Args:
            pattern: 查询模式列表，如
                [("Tesla", "CEO", "?person"), ("?person", "alma_mater", "?school")]

        Returns:
            所有满足模式的变量绑定列表，如
            [{"?person": "Elon Musk", "?school": "University of Pennsylvania"}]

        Example:
            >>> kg.query_pattern([
            ...     ("Tesla", "CEO", "?person"),
            ...     ("?person", "graduated_from", "?university")
            ... ])
            [{"?person": "Elon Musk", "?university": "UPenn"}]
        """
        if not pattern:
            return [{}]

        # 递归匹配模式
        return self._match_pattern(pattern, {})

    def _match_pattern(self, pattern: List[Tuple[str, str, str]],
                       bindings: Dict[str, str]) -> List[Dict[str, str]]:
        """递归模式匹配"""
        if not pattern:
            return [dict(bindings)]

        current = pattern[0]
        remaining = pattern[1:]
        results = []

        subj_raw, pred_raw, obj_raw = current

        # 替换已绑定的变量
        subj = bindings.get(subj_raw, subj_raw) if subj_raw.startswith("?") else subj_raw
        pred = bindings.get(pred_raw, pred_raw) if pred_raw.startswith("?") else pred_raw
        obj = bindings.get(obj_raw, obj_raw) if obj_raw.startswith("?") else obj_raw

        # 查找匹配的边
        candidate_edges = []

        if not subj.startswith("?") and not obj.startswith("?"):
            # 主语和宾语都已确定
            if self.graph.has_node(subj) and self.graph.has_node(obj):
                edge_data = self.graph.get_edge_data(subj, obj)
                if edge_data:
                    for key, data in edge_data.items():
                        if pred.startswith("?") or data.get("predicate") == pred:
                            candidate_edges.append((subj, data.get("predicate", ""), obj))

        elif not subj.startswith("?"):
            # 只有主语确定
            if self.graph.has_node(subj):
                for _, target, data in self.graph.out_edges(subj, data=True):
                    if pred.startswith("?") or data.get("predicate") == pred:
                        if obj.startswith("?") or target == obj:
                            candidate_edges.append((subj, data.get("predicate", ""), target))

        elif not obj.startswith("?"):
            # 只有宾语确定
            if self.graph.has_node(obj):
                for source_node, _, data in self.graph.in_edges(obj, data=True):
                    if pred.startswith("?") or data.get("predicate") == pred:
                        if subj.startswith("?") or source_node == subj:
                            candidate_edges.append((source_node, data.get("predicate", ""), obj))

        else:
            # 主语和宾语都是变量
            for u, v, data in self.graph.edges(data=True):
                if pred.startswith("?") or data.get("predicate") == pred:
                    candidate_edges.append((u, data.get("predicate", ""), v))

        # 对每个候选边，更新绑定并递归匹配剩余模式
        for edge_subj, edge_pred, edge_obj in candidate_edges:
            new_bindings = dict(bindings)
            conflict = False

            if subj_raw.startswith("?"):
                if subj_raw in new_bindings and new_bindings[subj_raw] != edge_subj:
                    conflict = True
                new_bindings[subj_raw] = edge_subj

            if pred_raw.startswith("?"):
                if pred_raw in new_bindings and new_bindings[pred_raw] != edge_pred:
                    conflict = True
                new_bindings[pred_raw] = edge_pred

            if obj_raw.startswith("?"):
                if obj_raw in new_bindings and new_bindings[obj_raw] != edge_obj:
                    conflict = True
                new_bindings[obj_raw] = edge_obj

            if not conflict:
                sub_results = self._match_pattern(remaining, new_bindings)
                results.extend(sub_results)

        return results

    # ----- 序列化 -----

    def save(self, path: str):
        """
        保存知识图谱

        保存结构：
        - {path}/graph.pkl: NetworkX 图结构
        - {path}/entity_index.json: 实体索引

        Args:
            path: 保存目录路径
        """
        os.makedirs(path, exist_ok=True)

        # 保存图结构
        graph_path = os.path.join(path, "graph.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(self.graph, f)

        # 保存实体索引
        index_data = {}
        for name, entity in self._entity_index.items():
            index_data[name] = entity.to_dict()

        index_path = os.path.join(path, "entity_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"知识图谱已保存到: {path} "
            f"({self.num_entities} 实体, {self.num_relations} 关系)"
        )

    @classmethod
    def load(cls, path: str) -> KnowledgeGraph:
        """
        从文件加载知识图谱

        Args:
            path: 保存目录路径

        Returns:
            加载的 KnowledgeGraph 实例
        """
        instance = cls()

        # 加载图结构
        graph_path = os.path.join(path, "graph.pkl")
        with open(graph_path, "rb") as f:
            instance.graph = pickle.load(f)

        # 加载实体索引
        index_path = os.path.join(path, "entity_index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            for name, entity_data in index_data.items():
                instance._entity_index[name] = Entity.from_dict(entity_data)

        logger.info(
            f"知识图谱已加载: {instance.num_entities} 实体, "
            f"{instance.num_relations} 关系"
        )

        return instance

    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        entity_types = {}
        for _, data in self.graph.nodes(data=True):
            etype = data.get("type", "unknown")
            entity_types[etype] = entity_types.get(etype, 0) + 1

        predicate_types = {}
        for _, _, data in self.graph.edges(data=True):
            pred = data.get("predicate", "unknown")
            predicate_types[pred] = predicate_types.get(pred, 0) + 1

        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "entity_types": entity_types,
            "predicate_types": predicate_types,
            "avg_degree": (
                sum(dict(self.graph.degree()).values()) / max(self.num_entities, 1)
            ),
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(entities={self.num_entities}, "
            f"relations={self.num_relations})"
        )
