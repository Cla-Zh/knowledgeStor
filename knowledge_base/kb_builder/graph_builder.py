"""
知识图谱构建器 (Knowledge Graph Builder)

负责从提取的实体和关系构建知识图谱（符号表示层）。

核心功能：
1. 从实体和关系列表构建 NetworkX 有向多重图
2. 实体去重和合并（同一实体的不同名称表述）
3. 关系归一化和去重
4. 图结构优化（移除孤立节点、合并弱连接分量等）
5. 图谱统计和可视化
6. 图谱的序列化和反序列化
"""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from data.models import Entity, EntityType, Fact, Question, Relation
from kb_builder.models import KnowledgeGraph, SearchResult
from utils.config import Config

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    知识图谱构建器

    将实体和关系转换为结构化的知识图谱（基于 NetworkX），
    支持多跳路径查询和符号推理。

    Usage:
        >>> builder = KnowledgeGraphBuilder()
        >>> entities = [Entity(name="Tesla", type="organization"), ...]
        >>> relations = [Relation(subject="Tesla", predicate="CEO", object="Elon Musk"), ...]
        >>> graph = builder.build_graph(entities, relations)
        >>> paths = graph.query_path("Tesla", "Pennsylvania", max_hops=3)
    """

    def __init__(
        self,
        min_confidence: Optional[float] = None,
        merge_similar_entities: bool = True,
        config: Optional[Config] = None,
    ):
        """
        初始化知识图谱构建器

        Args:
            min_confidence: 最低关系置信度阈值（低于此值的关系不加入图谱）
            merge_similar_entities: 是否合并名称相似的实体
            config: 配置对象
        """
        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else config.get("graph.min_confidence", 0.5)
        )
        self.merge_similar_entities = merge_similar_entities

        logger.info(
            f"KnowledgeGraphBuilder 初始化: "
            f"min_confidence={self.min_confidence}, "
            f"merge_similar={self.merge_similar_entities}"
        )

    # ==========================================
    # 核心构建方法
    # ==========================================

    def build_graph(self, entities: List[Entity],
                    relations: List[Relation]) -> KnowledgeGraph:
        """
        从实体和关系列表构建知识图谱

        流程：
        1. 实体去重和合并
        2. 关系过滤和归一化
        3. 构建图结构
        4. 后处理（移除孤立节点等）

        Args:
            entities: 实体列表
            relations: 关系列表

        Returns:
            构建好的 KnowledgeGraph 实例
        """
        logger.info(
            f"开始构建知识图谱: {len(entities)} 个实体, {len(relations)} 条关系"
        )

        # Step 1: 实体处理
        logger.info("Step 1/4: 实体去重和合并...")
        processed_entities = self._process_entities(entities)
        logger.info(f"  处理后实体数: {len(processed_entities)}")

        # Step 2: 关系处理
        logger.info("Step 2/4: 关系过滤和归一化...")
        processed_relations = self._process_relations(relations, processed_entities)
        logger.info(f"  处理后关系数: {len(processed_relations)}")

        # Step 3: 构建图结构
        logger.info("Step 3/4: 构建图结构...")
        kg = KnowledgeGraph()

        for entity in processed_entities:
            kg.add_entity(entity)

        for relation in processed_relations:
            kg.add_relation(relation)

        # Step 4: 后处理
        logger.info("Step 4/4: 图结构后处理...")
        self._post_process(kg)

        stats = kg.get_statistics()
        logger.info(
            f"知识图谱构建完成: "
            f"{stats['num_entities']} 实体, "
            f"{stats['num_relations']} 关系, "
            f"平均度={stats['avg_degree']:.2f}"
        )

        return kg

    def build_graph_from_questions(self, questions: List[Question]) -> KnowledgeGraph:
        """
        从 Question 列表直接构建知识图谱

        自动从问题中提取实体和关系。

        Args:
            questions: 问题列表

        Returns:
            构建好的 KnowledgeGraph 实例
        """
        logger.info(f"从 {len(questions)} 个问题构建知识图谱...")

        entities = []
        relations = []
        seen_entity_names: Set[str] = set()

        for question in questions:
            # 从上下文标题提取实体
            for title, sentences in question.context:
                if title and title not in seen_entity_names:
                    entity = Entity(
                        name=title,
                        type=self._infer_type_from_context(title, sentences),
                        source=f"context_{question.id}",
                    )
                    entities.append(entity)
                    seen_entity_names.add(title)

            # 从支撑事实中提取实体和关系
            for fact in question.supporting_facts:
                for ent_name in fact.entities:
                    if ent_name not in seen_entity_names:
                        entities.append(Entity(
                            name=ent_name,
                            source=f"fact_{question.id}",
                        ))
                        seen_entity_names.add(ent_name)
                relations.extend(fact.relations)

        return self.build_graph(entities, relations)

    def build_graph_from_facts(self, facts: List[Fact]) -> KnowledgeGraph:
        """
        从事实列表构建知识图谱

        Args:
            facts: 事实列表

        Returns:
            构建好的 KnowledgeGraph 实例
        """
        entities = []
        relations = []
        seen_names: Set[str] = set()

        for fact in facts:
            for ent_name in fact.entities:
                if ent_name not in seen_names:
                    entities.append(Entity(name=ent_name, source="fact"))
                    seen_names.add(ent_name)
            relations.extend(fact.relations)

        return self.build_graph(entities, relations)

    # ==========================================
    # 实体处理
    # ==========================================

    def _process_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        处理实体：去重、合并、归一化

        Args:
            entities: 原始实体列表

        Returns:
            处理后的实体列表
        """
        if not entities:
            return []

        # 按名称分组
        name_groups: Dict[str, List[Entity]] = defaultdict(list)
        for entity in entities:
            normalized_name = self._normalize_entity_name(entity.name)
            if normalized_name:
                name_groups[normalized_name].append(entity)

        # 合并同名实体
        merged_entities = []
        for name, group in name_groups.items():
            merged = self._merge_entity_group(group)
            merged_entities.append(merged)

        # 如果启用相似实体合并
        if self.merge_similar_entities:
            merged_entities = self._merge_similar_entities(merged_entities)

        return merged_entities

    def _normalize_entity_name(self, name: str) -> str:
        """
        归一化实体名称

        Args:
            name: 原始名称

        Returns:
            归一化后的名称
        """
        if not name:
            return ""
        # 去除首尾空白，统一空格
        normalized = " ".join(name.strip().split())
        return normalized

    def _merge_entity_group(self, group: List[Entity]) -> Entity:
        """
        合并一组同名实体

        策略：
        - 名称取第一个
        - 类型取最具体的（非 "other" 的优先）
        - 属性合并
        - 别名合并

        Args:
            group: 同名实体列表

        Returns:
            合并后的实体
        """
        if len(group) == 1:
            return group[0]

        base = group[0]

        # 选择最具体的类型
        best_type = EntityType.OTHER.value
        for entity in group:
            if entity.type != EntityType.OTHER.value:
                best_type = entity.type
                break

        # 合并属性
        merged_attrs = {}
        for entity in group:
            merged_attrs.update(entity.attributes)

        # 合并别名
        all_aliases = set()
        for entity in group:
            all_aliases.update(entity.aliases)
            # 把不同的原始名称也加入别名
            if entity.name != base.name:
                all_aliases.add(entity.name)
        all_aliases.discard(base.name)  # 移除与主名称相同的

        return Entity(
            id=base.id,
            name=base.name,
            type=best_type,
            attributes=merged_attrs,
            aliases=list(all_aliases),
            source=base.source,
        )

    def _merge_similar_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        合并名称相似的实体（简单启发式）

        当前策略：如果一个实体名是另一个的子串，且较短的名称长度 >= 3，
        则认为可能是同一实体。

        Args:
            entities: 实体列表

        Returns:
            合并后的实体列表
        """
        if len(entities) <= 1:
            return entities

        # 建立名称到实体的映射
        name_to_entity = {e.name.lower(): e for e in entities}
        merged_names: Set[str] = set()
        result = []

        sorted_entities = sorted(entities, key=lambda e: len(e.name), reverse=True)

        for entity in sorted_entities:
            name_lower = entity.name.lower()
            if name_lower in merged_names:
                continue

            # 检查是否有子串匹配
            for other in sorted_entities:
                other_lower = other.name.lower()
                if (other_lower != name_lower
                        and other_lower not in merged_names
                        and len(other_lower) >= 3
                        and other_lower in name_lower):
                    # 将短名称作为长名称的别名
                    if other.name not in entity.aliases:
                        entity.aliases.append(other.name)
                    merged_names.add(other_lower)

            result.append(entity)
            merged_names.add(name_lower)

        return result

    # ==========================================
    # 关系处理
    # ==========================================

    def _process_relations(self, relations: List[Relation],
                           entities: List[Entity]) -> List[Relation]:
        """
        处理关系：过滤、归一化、去重

        Args:
            relations: 原始关系列表
            entities: 处理后的实体列表

        Returns:
            处理后的关系列表
        """
        if not relations:
            return []

        # 构建实体名称集合（含别名），用于快速查找
        valid_names = set()
        for entity in entities:
            valid_names.add(entity.name)
            valid_names.update(entity.aliases)

        processed = []
        seen_triples: Set[Tuple[str, str, str]] = set()

        for relation in relations:
            # 过滤低置信度关系
            if relation.confidence < self.min_confidence:
                continue

            # 归一化主语和宾语
            subject = self._normalize_entity_name(relation.subject)
            obj = self._normalize_entity_name(relation.object)
            predicate = self._normalize_predicate(relation.predicate)

            # 过滤无效关系
            if not subject or not obj or not predicate:
                continue
            if subject == obj:  # 自环
                continue

            # 去重
            triple = (subject, predicate, obj)
            if triple in seen_triples:
                continue
            seen_triples.add(triple)

            processed.append(Relation(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=relation.confidence,
                source=relation.source,
                attributes=relation.attributes,
            ))

        return processed

    def _normalize_predicate(self, predicate: str) -> str:
        """
        归一化关系谓词

        Args:
            predicate: 原始谓词

        Returns:
            归一化后的谓词
        """
        if not predicate:
            return ""
        # 转小写，替换空格为下划线
        normalized = predicate.strip().lower().replace(" ", "_")
        return normalized

    # ==========================================
    # 后处理
    # ==========================================

    def _post_process(self, kg: KnowledgeGraph):
        """
        图结构后处理

        操作：
        1. 移除孤立节点（没有任何边的节点）
           - 仅当图中存在边时才移除孤立节点，否则保留所有实体用于实体名称检索
        2. 统计连通分量信息

        Args:
            kg: 知识图谱实例
        """
        # 移除孤立节点（仅在图中有边时执行，避免无关系数据集丢失全部实体）
        if kg.graph.number_of_edges() > 0:
            isolated = list(nx.isolates(kg.graph))
            if isolated:
                kg.graph.remove_nodes_from(isolated)
                logger.info(f"  移除了 {len(isolated)} 个孤立节点")
        else:
            logger.info(
                f"  图中无边（关系），保留全部 {kg.graph.number_of_nodes()} 个实体节点"
                f"（可用于实体名称检索）"
            )

        # 统计弱连通分量
        if kg.num_entities > 0:
            undirected = kg.graph.to_undirected()
            components = list(nx.connected_components(undirected))
            component_sizes = sorted(
                [len(c) for c in components], reverse=True
            )
            logger.info(
                f"  连通分量数: {len(components)}, "
                f"最大分量: {component_sizes[0] if component_sizes else 0} 实体"
            )

    # ==========================================
    # 图操作辅助方法
    # ==========================================

    def add_entity(self, graph: KnowledgeGraph, entity: Entity):
        """
        向已有图谱添加实体

        Args:
            graph: 知识图谱实例
            entity: 待添加的实体
        """
        graph.add_entity(entity)

    def add_relation(self, graph: KnowledgeGraph, relation: Relation):
        """
        向已有图谱添加关系

        Args:
            graph: 知识图谱实例
            relation: 待添加的关系
        """
        if relation.confidence >= self.min_confidence:
            graph.add_relation(relation)

    def merge_graphs(self, graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """
        合并多个知识图谱

        Args:
            graphs: 知识图谱列表

        Returns:
            合并后的知识图谱
        """
        if not graphs:
            return KnowledgeGraph()

        merged = KnowledgeGraph()

        for graph in graphs:
            # 合并节点
            for name, data in graph.graph.nodes(data=True):
                if not merged.graph.has_node(name):
                    merged.graph.add_node(name, **data)
                else:
                    # 更新已有节点的属性
                    existing_attrs = merged.graph.nodes[name].get("attributes", {})
                    new_attrs = data.get("attributes", {})
                    existing_attrs.update(new_attrs)
                    merged.graph.nodes[name]["attributes"] = existing_attrs

            # 合并边
            for u, v, data in graph.graph.edges(data=True):
                merged.graph.add_edge(u, v, **data)

            # 合并实体索引
            merged._entity_index.update(graph._entity_index)

        logger.info(
            f"合并了 {len(graphs)} 个图谱: "
            f"{merged.num_entities} 实体, {merged.num_relations} 关系"
        )

        return merged

    # ==========================================
    # 图谱查询便捷方法
    # ==========================================

    def search_entity(self, graph: KnowledgeGraph,
                      query: str, top_k: int = 5) -> List[SearchResult]:
        """
        在图谱中搜索实体（基于名称模糊匹配）

        Args:
            graph: 知识图谱
            query: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            搜索结果列表
        """
        query_lower = query.lower().strip()
        results = []

        for name, data in graph.graph.nodes(data=True):
            name_lower = name.lower()
            score = 0.0

            # 完全匹配
            if name_lower == query_lower:
                score = 1.0
            # 包含匹配
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
                source="graph",
            ))

        # 按分数排序
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ==========================================
    # 持久化
    # ==========================================

    def save_graph(self, graph: KnowledgeGraph, path: str):
        """
        保存知识图谱

        Args:
            graph: 知识图谱实例
            path: 保存目录路径
        """
        graph.save(path)
        logger.info(f"知识图谱已保存到: {path}")

    def load_graph(self, path: str) -> KnowledgeGraph:
        """
        加载知识图谱

        Args:
            path: 保存目录路径

        Returns:
            加载的 KnowledgeGraph 实例
        """
        graph = KnowledgeGraph.load(path)
        logger.info(
            f"知识图谱已加载: {graph.num_entities} 实体, "
            f"{graph.num_relations} 关系"
        )
        return graph

    # ==========================================
    # 为多个专家构建图谱
    # ==========================================

    def build_expert_graphs(
        self,
        expert_data: Dict[str, Tuple[List[Entity], List[Relation]]],
        base_path: str,
    ) -> Dict[str, KnowledgeGraph]:
        """
        为多个专家分别构建知识图谱

        Args:
            expert_data: {expert_id: (entities, relations)} 专家数据字典
            base_path: 保存的根目录

        Returns:
            {expert_id: KnowledgeGraph} 图谱字典
        """
        graphs = {}

        for expert_id, (entities, relations) in expert_data.items():
            logger.info(
                f"构建专家 '{expert_id}' 的知识图谱 "
                f"({len(entities)} 实体, {len(relations)} 关系)..."
            )

            # 构建图谱
            graph = self.build_graph(entities, relations)

            # 保存图谱
            expert_path = os.path.join(base_path, expert_id)
            self.save_graph(graph, expert_path)

            graphs[expert_id] = graph

        logger.info(f"所有专家图谱构建完成: {len(graphs)} 个专家")
        return graphs

    # ==========================================
    # 统计和分析
    # ==========================================

    def analyze_graph(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        分析知识图谱的结构特征

        Args:
            graph: 知识图谱

        Returns:
            分析报告字典
        """
        stats = graph.get_statistics()

        g = graph.graph

        # 度分布
        in_degrees = dict(g.in_degree())
        out_degrees = dict(g.out_degree())

        # 高度数节点（Hub 节点）
        top_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        top_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]

        # 关系类型分布
        predicate_counter = Counter()
        for _, _, data in g.edges(data=True):
            predicate_counter[data.get("predicate", "unknown")] += 1

        stats.update({
            "top_in_degree_entities": [
                {"name": name, "in_degree": deg} for name, deg in top_in
            ],
            "top_out_degree_entities": [
                {"name": name, "out_degree": deg} for name, deg in top_out
            ],
            "predicate_distribution": dict(predicate_counter.most_common(20)),
            "density": nx.density(g) if g.number_of_nodes() > 0 else 0,
        })

        return stats

    # ==========================================
    # 辅助方法
    # ==========================================

    @staticmethod
    def _infer_type_from_context(title: str, sentences: List[str]) -> str:
        """从上下文推断实体类型"""
        text = (title + " " + " ".join(sentences if isinstance(sentences, list) else [sentences])).lower()

        type_hints = {
            EntityType.PERSON.value: [
                "born", "died", "actor", "actress", "politician",
                "president", "singer", "writer",
            ],
            EntityType.LOCATION.value: [
                "city", "country", "located", "capital", "state", "region",
            ],
            EntityType.ORGANIZATION.value: [
                "company", "university", "founded", "corporation", "school",
            ],
            EntityType.WORK.value: [
                "film", "movie", "book", "album", "directed", "starring",
            ],
        }

        scores = {}
        for etype, keywords in type_hints.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[etype] = score

        if scores:
            return max(scores, key=scores.get)
        return EntityType.OTHER.value

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraphBuilder(min_confidence={self.min_confidence}, "
            f"merge_similar={self.merge_similar_entities})"
        )
