"""
专家领域识别器 (Expert Identifier)

自动从数据集中识别专家领域，支持三种识别方法：
1. clustering: 基于实体类型聚类（默认，推荐）
2. lda: 基于 LDA 主题模型
3. entity_based: 基于预定义实体类型规则

核心流程：
1. 从数据集中提取特征（实体类型、关键词、文本内容）
2. 使用选定算法将数据聚类为多个领域
3. 为每个领域生成 ExpertDomain 描述
4. 将数据分配到各专家领域
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from data.models import Entity, EntityType, Question, Relation
from data.loader import DatasetLoader
from kb_builder.models import ExpertDomain
from utils.config import Config

logger = logging.getLogger(__name__)


# ==============================================================================
# 预定义实体类型分组（用于 entity_based 方法）
# ==============================================================================

DEFAULT_ENTITY_GROUPS = {
    "person": {
        "name": "人物领域",
        "description": "涵盖人物相关知识：演员、政治家、运动员、科学家等",
        "entity_types": ["person", "actor", "politician", "athlete", "scientist",
                         "writer", "artist", "musician", "director"],
        "keywords": ["born", "died", "career", "award", "education", "married",
                     "children", "nationality", "occupation", "age"],
    },
    "location": {
        "name": "地理领域",
        "description": "涵盖地理相关知识：城市、国家、地标、地区等",
        "entity_types": ["location", "city", "country", "landmark", "region",
                         "state", "continent", "island", "mountain", "river"],
        "keywords": ["located", "capital", "population", "area", "border",
                     "latitude", "longitude", "climate", "continent"],
    },
    "organization": {
        "name": "组织领域",
        "description": "涵盖组织相关知识：公司、大学、政府机构、团体等",
        "entity_types": ["organization", "company", "university", "school",
                         "government", "agency", "team", "club", "party"],
        "keywords": ["founded", "headquarters", "CEO", "revenue", "employees",
                     "member", "established", "president", "chairman"],
    },
    "work": {
        "name": "作品领域",
        "description": "涵盖作品相关知识：电影、书籍、音乐、游戏等",
        "entity_types": ["work", "film", "movie", "book", "novel", "album",
                         "song", "show", "series", "game"],
        "keywords": ["directed", "starring", "released", "published", "written",
                     "produced", "genre", "award", "nominated", "rating"],
    },
    "event": {
        "name": "事件领域",
        "description": "涵盖事件相关知识：战争、选举、比赛、节日等",
        "entity_types": ["event", "war", "battle", "election", "tournament",
                         "championship", "festival", "ceremony"],
        "keywords": ["occurred", "started", "ended", "participants", "winner",
                     "result", "casualties", "location", "date"],
    },
}


class ExpertIdentifier:
    """
    专家领域自动识别器

    从数据集中自动发现领域边界，将数据划分为多个专家领域。

    Usage:
        >>> identifier = ExpertIdentifier(n_experts=3, method='clustering')
        >>> questions = DatasetLoader().load_2wiki("data/raw/train.json")
        >>> domains = identifier.identify_domains(questions)
        >>> expert_data = identifier.assign_data_to_experts(questions, domains)
    """

    def __init__(
        self,
        n_experts: Optional[int] = None,
        method: str = "clustering",
        config: Optional[Config] = None,
    ):
        """
        初始化专家领域识别器

        Args:
            n_experts: 专家数量，None 表示自动确定
            method: 识别方法 - 'clustering', 'lda', 'entity_based'
            config: 配置对象
        """
        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self.n_experts = n_experts or config.get("system.n_experts", None)
        self.method = method or config.get("system.expert_identification_method", "clustering")
        self.random_seed = config.get("system.random_seed", 42)

        # 内部状态
        self._loader = DatasetLoader()
        self._domains: List[ExpertDomain] = []
        self._assignment: Dict[str, List[int]] = {}  # expert_id -> question indices

        logger.info(
            f"ExpertIdentifier 初始化: method={self.method}, "
            f"n_experts={self.n_experts or 'auto'}"
        )

    # ==========================================
    # 主接口
    # ==========================================

    def identify_domains(self, questions: List[Question]) -> List[ExpertDomain]:
        """
        从问题列表中识别专家领域

        Args:
            questions: Question 对象列表

        Returns:
            识别出的专家领域列表
        """
        if not questions:
            logger.warning("问题列表为空，无法识别专家领域")
            return []

        logger.info(f"开始识别专家领域: {len(questions)} 个问题, 方法={self.method}")

        if self.method == "clustering":
            domains = self._identify_by_clustering(questions)
        elif self.method == "lda":
            domains = self._identify_by_lda(questions)
        elif self.method == "entity_based":
            domains = self._identify_by_entity_type(questions)
        else:
            raise ValueError(f"不支持的识别方法: {self.method}")

        self._domains = domains

        logger.info(f"识别完成: {len(domains)} 个专家领域")
        for domain in domains:
            logger.info(
                f"  {domain.id}: {domain.name} "
                f"(types={domain.entity_types}, count={domain.data_count})"
            )

        return domains

    def assign_data_to_experts(
        self,
        questions: List[Question],
        domains: List[ExpertDomain],
    ) -> Dict[str, List[Question]]:
        """
        将数据分配给不同的专家库

        每个问题根据其涉及的实体类型和关键词分配到最匹配的专家。
        一个问题可能被分配到多个专家（如果涉及多个领域）。

        Args:
            questions: 问题列表
            domains: 专家领域列表

        Returns:
            {expert_id: [Question, ...]} 分配结果
        """
        if not domains:
            logger.warning("专家领域为空，所有数据分配到默认专家")
            return {"expert_default": questions}

        logger.info(f"开始将 {len(questions)} 个问题分配到 {len(domains)} 个专家...")

        assignment: Dict[str, List[Question]] = {d.id: [] for d in domains}

        for question in questions:
            # 计算该问题与各领域的匹配度
            scores = self._compute_domain_scores(question, domains)

            # 分配到得分最高的领域
            best_domain_id = max(scores, key=scores.get)
            assignment[best_domain_id].append(question)

            # 如果是多跳问题，也可能需要分配到次优领域
            if question.is_multi_hop:
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for domain_id, score in sorted_scores[1:]:
                    # 次优领域得分达到最优的 60% 以上，也分配
                    if score >= sorted_scores[0][1] * 0.6 and score > 0:
                        if question not in assignment[domain_id]:
                            assignment[domain_id].append(question)

        # 更新领域的数据量
        for domain in domains:
            domain.data_count = len(assignment[domain.id])

        # 日志
        for domain_id, qs in assignment.items():
            logger.info(f"  {domain_id}: {len(qs)} 个问题")

        return assignment

    # ==========================================
    # 方法 A: 基于实体类型聚类
    # ==========================================

    def _identify_by_clustering(self, questions: List[Question]) -> List[ExpertDomain]:
        """
        基于实体类型聚类识别专家领域

        流程：
        1. 为每个问题构建特征向量（实体类型分布 + 关键词 TF-IDF）
        2. 使用 K-Means 聚类
        3. 分析每个簇的特征，生成领域描述

        Args:
            questions: 问题列表

        Returns:
            专家领域列表
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        logger.info("使用聚类方法识别专家领域...")

        # Step 1: 提取每个问题的文本特征
        question_texts = []
        for q in questions:
            # 组合问题文本 + 上下文标题 + 支撑事实文本
            parts = [q.text]
            for title, _ in q.context:
                parts.append(title)
            for fact in q.supporting_facts:
                if fact.text:
                    parts.append(fact.text)
            question_texts.append(" ".join(parts))

        # Step 2: TF-IDF 特征提取
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(question_texts)

        # Step 3: 确定聚类数量
        n_clusters = self.n_experts
        if n_clusters is None:
            n_clusters = self._find_optimal_k(tfidf_matrix, max_k=10)
            logger.info(f"自动确定最佳专家数量: {n_clusters}")

        # Step 4: K-Means 聚类
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_seed,
            n_init=10,
            max_iter=300,
        )
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        # Step 5: 分析每个簇，生成领域描述
        feature_names = vectorizer.get_feature_names_out()
        domains = []

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_questions = [questions[i] for i in cluster_indices]

            # 分析实体类型分布
            entity_types = self._analyze_entity_types(cluster_questions)

            # 提取簇的关键词
            cluster_center = kmeans.cluster_centers_[cluster_id]
            top_keyword_indices = cluster_center.argsort()[-15:][::-1]
            keywords = [feature_names[i] for i in top_keyword_indices]

            # 推断领域名称
            domain_name = self._infer_domain_name(entity_types, keywords)

            # 计算簇中心（用于路由）
            cluster_tfidf = tfidf_matrix[cluster_mask].toarray()
            centroid = np.mean(cluster_tfidf, axis=0).astype(np.float32)

            domain = ExpertDomain(
                id=f"expert_{cluster_id}",
                name=domain_name,
                description=f"自动识别的领域: {domain_name}，包含 {len(cluster_questions)} 条数据",
                entity_types=entity_types[:5],  # 取前 5 个主要类型
                keywords=keywords[:10],
                data_count=len(cluster_questions),
                centroid=centroid,
            )
            domains.append(domain)

        return domains

    def _find_optimal_k(self, features, max_k: int = 10) -> int:
        """
        使用轮廓系数自动确定最佳聚类数

        Args:
            features: 特征矩阵
            max_k: 最大聚类数

        Returns:
            最佳聚类数
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n_samples = features.shape[0]
        max_k = min(max_k, n_samples - 1, 10)

        if max_k < 2:
            return 2

        best_k = 3  # 默认
        best_score = -1

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=5)
            labels = kmeans.fit_predict(features)

            # 轮廓系数：-1 到 1，越高越好
            try:
                score = silhouette_score(features, labels, sample_size=min(5000, n_samples))
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        logger.info(f"轮廓系数最优 k={best_k}, score={best_score:.4f}")
        return best_k

    # ==========================================
    # 方法 B: 基于 LDA 主题模型
    # ==========================================

    def _identify_by_lda(self, questions: List[Question]) -> List[ExpertDomain]:
        """
        基于 LDA 主题模型识别专家领域

        流程：
        1. 将所有文档进行 LDA 主题建模
        2. 每个主题对应一个专家领域
        3. 根据主题分布分配数据

        Args:
            questions: 问题列表

        Returns:
            专家领域列表
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        logger.info("使用 LDA 主题模型识别专家领域...")

        # Step 1: 构建文档
        documents = []
        for q in questions:
            parts = [q.text]
            for title, _ in q.context:
                parts.append(title)
            for fact in q.supporting_facts:
                if fact.text:
                    parts.append(fact.text)
            documents.append(" ".join(parts))

        # Step 2: 词袋模型
        count_vectorizer = CountVectorizer(
            max_features=5000,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        count_matrix = count_vectorizer.fit_transform(documents)

        # Step 3: LDA 建模
        n_topics = self.n_experts or 5
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=self.random_seed,
            max_iter=50,
            learning_method="online",
            batch_size=128,
        )
        doc_topic_matrix = lda.fit_transform(count_matrix)

        # Step 4: 分析每个主题
        feature_names = count_vectorizer.get_feature_names_out()
        domains = []

        for topic_id in range(n_topics):
            # 获取主题的 top 关键词
            topic_word_weights = lda.components_[topic_id]
            top_word_indices = topic_word_weights.argsort()[-15:][::-1]
            keywords = [feature_names[i] for i in top_word_indices]

            # 找出属于该主题的问题
            topic_questions = [
                questions[i]
                for i in range(len(questions))
                if doc_topic_matrix[i].argmax() == topic_id
            ]

            # 分析实体类型
            entity_types = self._analyze_entity_types(topic_questions)

            # 推断领域名称
            domain_name = self._infer_domain_name(entity_types, keywords)

            domain = ExpertDomain(
                id=f"expert_{topic_id}",
                name=domain_name,
                description=f"LDA 主题 {topic_id}: {domain_name}",
                entity_types=entity_types[:5],
                keywords=keywords[:10],
                data_count=len(topic_questions),
            )
            domains.append(domain)

        return domains

    # ==========================================
    # 方法 C: 基于预定义实体类型规则
    # ==========================================

    def _identify_by_entity_type(self, questions: List[Question]) -> List[ExpertDomain]:
        """
        基于预定义实体类型规则识别专家领域

        使用 DEFAULT_ENTITY_GROUPS 中的预定义分组，
        统计数据集中各分组的出现频率，只保留有实际数据的分组。

        Args:
            questions: 问题列表

        Returns:
            专家领域列表
        """
        logger.info("使用实体类型规则识别专家领域...")

        # 统计每个预定义领域的匹配问题数
        domain_counts: Dict[str, int] = {}
        for group_id, group_info in DEFAULT_ENTITY_GROUPS.items():
            count = 0
            for q in questions:
                if self._question_matches_group(q, group_info):
                    count += 1
            domain_counts[group_id] = count

        # 只保留有数据的领域
        domains = []
        expert_idx = 0
        for group_id, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            if count == 0:
                continue
            group_info = DEFAULT_ENTITY_GROUPS[group_id]
            domain = ExpertDomain(
                id=f"expert_{expert_idx}",
                name=group_info["name"],
                description=group_info["description"],
                entity_types=group_info["entity_types"],
                keywords=group_info["keywords"],
                data_count=count,
            )
            domains.append(domain)
            expert_idx += 1

        # 限制专家数量
        if self.n_experts and len(domains) > self.n_experts:
            domains = domains[:self.n_experts]

        return domains

    def _question_matches_group(self, question: Question,
                                 group_info: Dict) -> bool:
        """检查问题是否匹配某个实体分组"""
        # 构建问题文本
        text_parts = [question.text.lower()]
        for title, _ in question.context:
            text_parts.append(title.lower())
        for fact in question.supporting_facts:
            text_parts.extend(e.lower() for e in fact.entities)
        full_text = " ".join(text_parts)

        # 检查关键词匹配
        keyword_matches = sum(
            1 for kw in group_info["keywords"] if kw.lower() in full_text
        )
        # 检查实体类型匹配
        type_matches = 0
        for fact in question.supporting_facts:
            for rel in fact.relations:
                rel_text = rel.predicate.lower()
                for kw in group_info["keywords"]:
                    if kw.lower() in rel_text:
                        type_matches += 1

        return (keyword_matches >= 2) or (type_matches >= 1)

    # ==========================================
    # 辅助方法
    # ==========================================

    def _compute_domain_scores(self, question: Question,
                                domains: List[ExpertDomain]) -> Dict[str, float]:
        """
        计算一个问题与各专家领域的匹配度

        融合三种信号：
        1. 实体类型匹配
        2. 关键词匹配
        3. 领域中心向量相似度（如果有 centroid）

        Args:
            question: 问题对象
            domains: 专家领域列表

        Returns:
            {domain_id: score} 匹配度字典
        """
        # 提取问题特征
        q_text = question.text.lower()
        q_entities = set()
        q_relations = set()

        for fact in question.supporting_facts:
            q_entities.update(e.lower() for e in fact.entities)
            q_relations.update(r.predicate.lower() for r in fact.relations)

        for title, _ in question.context:
            q_entities.add(title.lower())

        full_text = q_text + " " + " ".join(q_entities) + " " + " ".join(q_relations)

        scores = {}
        for domain in domains:
            score = 0.0

            # 1. 关键词匹配 (权重 0.5)
            keyword_hits = sum(
                1 for kw in domain.keywords if kw.lower() in full_text
            )
            score += 0.5 * (keyword_hits / max(len(domain.keywords), 1))

            # 2. 实体类型匹配 (权重 0.3)
            type_hits = sum(
                1 for et in domain.entity_types if et.lower() in full_text
            )
            score += 0.3 * (type_hits / max(len(domain.entity_types), 1))

            # 3. 关系谓词匹配 (权重 0.2)
            for rel_pred in q_relations:
                for kw in domain.keywords:
                    if kw.lower() in rel_pred:
                        score += 0.2
                        break

            scores[domain.id] = score

        # 如果所有得分为 0，均匀分配
        if all(s == 0 for s in scores.values()):
            for domain in domains:
                scores[domain.id] = 1.0 / len(domains)

        return scores

    def _analyze_entity_types(self, questions: List[Question]) -> List[str]:
        """
        分析一组问题中的实体类型分布

        Args:
            questions: 问题列表

        Returns:
            按频率排序的实体类型列表
        """
        type_counter: Counter = Counter()

        for q in questions:
            # 从支撑事实的关系谓词中推断实体类型
            for fact in q.supporting_facts:
                for rel in fact.relations:
                    inferred_type = self._infer_type_from_relation(rel.predicate)
                    if inferred_type:
                        type_counter[inferred_type] += 1

            # 从上下文内容推断
            for title, sentences in q.context:
                text = title + " " + (
                    " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                )
                for etype, hints in _ENTITY_TYPE_HINTS.items():
                    if any(h in text.lower() for h in hints):
                        type_counter[etype] += 1

        # 返回按频率排序的类型
        return [t for t, _ in type_counter.most_common()]

    @staticmethod
    def _infer_type_from_relation(predicate: str) -> Optional[str]:
        """从关系谓词推断实体类型"""
        pred_lower = predicate.lower()

        person_hints = ["born", "died", "spouse", "child", "parent", "nationality",
                        "occupation", "educated", "award"]
        location_hints = ["located", "capital", "country", "state", "city", "region"]
        org_hints = ["founded", "CEO", "headquarters", "member", "employee"]
        work_hints = ["directed", "starring", "released", "published", "genre"]

        if any(h in pred_lower for h in person_hints):
            return "person"
        if any(h in pred_lower for h in location_hints):
            return "location"
        if any(h in pred_lower for h in org_hints):
            return "organization"
        if any(h in pred_lower for h in work_hints):
            return "work"

        return None

    @staticmethod
    def _infer_domain_name(entity_types: List[str], keywords: List[str]) -> str:
        """
        根据实体类型和关键词推断领域名称

        Args:
            entity_types: 实体类型列表
            keywords: 关键词列表

        Returns:
            领域名称
        """
        if not entity_types:
            # 从关键词推断
            keyword_set = set(kw.lower() for kw in keywords[:10])
            for group_id, group_info in DEFAULT_ENTITY_GROUPS.items():
                group_kws = set(kw.lower() for kw in group_info["keywords"])
                overlap = keyword_set & group_kws
                if len(overlap) >= 2:
                    return group_info["name"]
            return "通用领域"

        primary_type = entity_types[0].lower()

        type_name_map = {
            "person": "人物领域",
            "location": "地理领域",
            "organization": "组织领域",
            "work": "作品领域",
            "event": "事件领域",
            "date": "时间领域",
            "number": "数值领域",
        }

        return type_name_map.get(primary_type, f"{primary_type}领域")

    def get_domains(self) -> List[ExpertDomain]:
        """获取已识别的领域列表"""
        return self._domains

    def __repr__(self) -> str:
        return (
            f"ExpertIdentifier(method='{self.method}', "
            f"n_experts={self.n_experts or 'auto'}, "
            f"domains={len(self._domains)})"
        )


# ==============================================================================
# 模块级辅助数据
# ==============================================================================

_ENTITY_TYPE_HINTS = {
    "person": ["born", "died", "actor", "actress", "politician", "president",
               "singer", "writer", "artist", "athlete", "player", "coach"],
    "location": ["city", "country", "capital", "located", "state", "region",
                 "province", "continent", "island", "mountain", "river"],
    "organization": ["company", "university", "school", "founded", "headquarters",
                     "corporation", "team", "club", "agency"],
    "work": ["film", "movie", "book", "novel", "album", "song", "show",
             "series", "directed", "starring", "released"],
    "event": ["war", "battle", "election", "tournament", "championship",
              "festival", "ceremony", "revolution"],
}
