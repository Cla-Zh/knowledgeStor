"""
语义解析器 (Semantic Parser)

将自然语言问题解析为结构化表示，是神经-符号转换层的核心组件。

功能：
1. 识别问题类型（单跳/多跳/比较等）
2. 提取问题中的实体
3. 识别问题中涉及的关系
4. 将问题分解为多跳查询结构
5. 生成知识图谱查询和向量检索查询

使用两级策略：
- 规则/模板：处理常见问题模式（快速、精确）
- 可选通过 Chat API 增强：处理复杂/非模板问题（灵活、通用）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from data.models import QuestionType
from utils.config import Config

logger = logging.getLogger(__name__)


# ==============================================================================
# 数据模型
# ==============================================================================


@dataclass
class HopQuery:
    """
    单跳查询描述

    Attributes:
        hop_id: 跳数编号（从 1 开始）
        intent: 查询意图 - 'find_entity', 'find_attribute', 'find_relation', 'compare'
        target: 查询目标描述
        expected_type: 预期返回的实体类型
        depends_on: 依赖的前序 hop_id（用于链式推理）
    """

    hop_id: int = 0
    intent: str = "find_entity"
    target: str = ""
    expected_type: str = ""
    depends_on: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hop_id": self.hop_id,
            "intent": self.intent,
            "target": self.target,
            "expected_type": self.expected_type,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HopQuery:
        return cls(
            hop_id=data.get("hop_id", 0),
            intent=data.get("intent", "find_entity"),
            target=data.get("target", ""),
            expected_type=data.get("expected_type", ""),
            depends_on=data.get("depends_on"),
        )


@dataclass
class StructuredQuery:
    """
    结构化查询

    将自然语言问题解析为可执行的结构化表示。

    Attributes:
        original_question: 原始问题文本
        query_type: 查询类型（single_hop, multi_hop, comparison 等）
        hops: 各跳查询列表
        entities: 问题中识别出的实体
        relations: 问题中识别出的关系
        graph_query: 生成的知识图谱查询（模式匹配格式）
        vector_query: 生成的向量检索查询文本
        metadata: 附加信息
    """

    original_question: str = ""
    query_type: str = QuestionType.UNKNOWN.value
    hops: List[HopQuery] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    graph_query: Optional[Dict[str, Any]] = None
    vector_query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_hops(self) -> int:
        return len(self.hops)

    @property
    def is_multi_hop(self) -> bool:
        return len(self.hops) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_question": self.original_question,
            "query_type": self.query_type,
            "hops": [h.to_dict() for h in self.hops],
            "entities": self.entities,
            "relations": self.relations,
            "graph_query": self.graph_query,
            "vector_query": self.vector_query,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StructuredQuery:
        return cls(
            original_question=data.get("original_question", ""),
            query_type=data.get("query_type", QuestionType.UNKNOWN.value),
            hops=[HopQuery.from_dict(h) for h in data.get("hops", [])],
            entities=data.get("entities", []),
            relations=data.get("relations", []),
            graph_query=data.get("graph_query"),
            vector_query=data.get("vector_query", ""),
            metadata=data.get("metadata", {}),
        )


# ==============================================================================
# 语义解析器
# ==============================================================================


# 常用疑问词模式
_WH_PATTERNS = {
    "who": {"intent": "find_entity", "expected_type": "person"},
    "where": {"intent": "find_entity", "expected_type": "location"},
    "when": {"intent": "find_attribute", "expected_type": "date"},
    "what": {"intent": "find_entity", "expected_type": ""},
    "which": {"intent": "find_entity", "expected_type": ""},
    "how many": {"intent": "find_attribute", "expected_type": "number"},
    "how much": {"intent": "find_attribute", "expected_type": "number"},
    "how old": {"intent": "find_attribute", "expected_type": "number"},
}

# 比较类问题关键词
_COMPARISON_KEYWORDS = [
    "more", "less", "greater", "smaller", "taller", "shorter",
    "older", "younger", "higher", "lower", "bigger", "longer",
    "both", "between", "compare", "difference", "same",
    "which one", "who is more", "which is",
]

# 常见关系模式：(正则, 关系谓词)
_RELATION_PATTERNS = [
    (r"(?:CEO|chief executive|head) of", "CEO"),
    (r"(?:founder|founded|created) (?:of|by)", "founder"),
    (r"(?:born|birth) (?:in|at|on)", "born_in"),
    (r"(?:died|death) (?:in|at|on)", "died_in"),
    (r"(?:capital|capital city) of", "capital_of"),
    (r"(?:located|situated|based) (?:in|at)", "located_in"),
    (r"(?:married|spouse|wife|husband) (?:of|to)", "spouse"),
    (r"(?:directed|director) (?:of|by)", "directed_by"),
    (r"(?:starred|starring|acted) (?:in|by)", "starred_in"),
    (r"(?:written|author|wrote) (?:of|by)", "author_of"),
    (r"(?:graduated|attended|studied|alma mater) (?:from|at)", "educated_at"),
    (r"(?:member|belongs) (?:of|to)", "member_of"),
    (r"(?:parent|father|mother) of", "parent_of"),
    (r"(?:child|son|daughter) of", "child_of"),
    (r"(?:country|nation|nationality) of", "country_of"),
    (r"(?:population|inhabitants) of", "population"),
    (r"(?:released|published|came out) (?:in|on)", "released_in"),
]


class SemanticParser:
    """
    语义解析器

    将自然语言问题解析为结构化查询，支持：
    - 实体提取
    - 关系识别
    - 问题类型判断
    - 多跳查询分解
    - 知识图谱查询生成
    - 向量检索查询优化

    Usage:
        >>> parser = SemanticParser()
        >>> sq = parser.parse("Who is the CEO of the company that made the iPhone?")
        >>> print(sq.query_type)   # "multi_hop"
        >>> print(sq.hops)         # [HopQuery(1, "find_entity", ...), HopQuery(2, ...)]
        >>> print(sq.graph_query)  # {"pattern": [...], "return": [...]}
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化语义解析器

        Args:
            config: 配置对象
        """
        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self._config = config

        logger.info("SemanticParser 初始化完成")

    # ==========================================
    # 主接口
    # ==========================================

    def parse(self, question: str) -> StructuredQuery:
        """
        解析问题为结构化查询

        流程：
        1. 提取实体和关系
        2. 判断问题类型
        3. 分解为多跳查询
        4. 生成图谱查询和向量查询

        Args:
            question: 自然语言问题

        Returns:
            结构化查询对象
        """
        question = question.strip()
        if not question:
            return StructuredQuery(original_question=question)

        logger.debug(f"解析问题: {question}")

        # Step 1: 提取实体和关系
        entities = self.extract_entities(question)
        relations = self.identify_relations(question)

        # Step 2: 判断问题类型
        query_type = self._classify_question(question, entities, relations)

        # Step 3: 分解为多跳查询
        hops = self._decompose_question(question, entities, relations, query_type)

        # Step 4: 生成查询
        graph_query = self._generate_graph_query(entities, relations, hops)
        vector_query = self._generate_vector_query(question, entities, relations)

        structured_query = StructuredQuery(
            original_question=question,
            query_type=query_type,
            hops=hops,
            entities=entities,
            relations=relations,
            graph_query=graph_query,
            vector_query=vector_query,
            metadata={
                "wh_word": self._extract_wh_word(question),
            },
        )

        logger.debug(
            f"解析结果: type={query_type}, entities={entities}, "
            f"relations={relations}, hops={len(hops)}"
        )
        return structured_query

    # ==========================================
    # 实体提取
    # ==========================================

    def extract_entities(self, question: str) -> List[str]:
        """
        从问题中提取实体（命名实体识别）

        策略组合：
        1. 规则：引号中的内容、大写单词序列
        2. SpaCy NER（如果可用）
        3. 简单启发式：介词 "of" 后的名词短语

        Args:
            question: 问题文本

        Returns:
            实体名称列表
        """
        entities = []
        seen = set()

        # 策略 1: 引号中的内容
        quoted = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]', question)
        for q in quoted:
            q = q.strip()
            if q and q not in seen:
                entities.append(q)
                seen.add(q)

        # 策略 2: 大写单词序列（英文专有名词）
        # 匹配连续的以大写字母开头的单词（至少 1 个）
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for match in re.finditer(cap_pattern, question):
            name = match.group(1).strip()
            # 过滤掉句首单词（可能只是普通大写）
            if match.start() > 0 and name not in seen and len(name) > 1:
                entities.append(name)
                seen.add(name)

        # 策略 3: SpaCy NER（如果可用）
        spacy_entities = self._spacy_extract_entities(question)
        for ent in spacy_entities:
            if ent not in seen:
                entities.append(ent)
                seen.add(ent)

        # 策略 4: "of X" 模式提取
        of_pattern = r'\bof\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?)(?:\?|,|\.|$|\s+(?:and|or|in|at|on|is|was|who|that|which))'
        for match in re.finditer(of_pattern, question):
            name = match.group(1).strip()
            if name and name not in seen and len(name) > 1:
                entities.append(name)
                seen.add(name)

        return entities

    def _spacy_extract_entities(self, text: str) -> List[str]:
        """使用 SpaCy 提取命名实体"""
        try:
            import spacy
            # 延迟加载 SpaCy 模型
            if not hasattr(self, '_nlp'):
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.debug("SpaCy 模型 en_core_web_sm 未安装，跳过 NER")
                    self._nlp = None
                    return []

            if self._nlp is None:
                return []

            doc = self._nlp(text)
            entities = []
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC",
                                   "NORP", "PRODUCT", "EVENT", "WORK_OF_ART"):
                    entities.append(ent.text)
            return entities

        except ImportError:
            return []

    # ==========================================
    # 关系识别
    # ==========================================

    def identify_relations(self, question: str) -> List[str]:
        """
        识别问题中涉及的关系

        使用预定义的关系模式进行匹配。

        Args:
            question: 问题文本

        Returns:
            关系谓词列表
        """
        question_lower = question.lower()
        relations = []
        seen = set()

        for pattern, relation in _RELATION_PATTERNS:
            if re.search(pattern, question_lower) and relation not in seen:
                relations.append(relation)
                seen.add(relation)

        return relations

    # ==========================================
    # 问题类型分类
    # ==========================================

    def _classify_question(self, question: str, entities: List[str],
                            relations: List[str]) -> str:
        """
        判断问题类型

        Args:
            question: 问题文本
            entities: 提取的实体
            relations: 识别的关系

        Returns:
            问题类型字符串
        """
        q_lower = question.lower()

        # 比较类问题
        if any(kw in q_lower for kw in _COMPARISON_KEYWORDS):
            return QuestionType.COMPARISON.value

        # Yes/No 问题
        if q_lower.startswith(("is ", "are ", "was ", "were ", "do ", "does ",
                                "did ", "has ", "have ", "can ", "could ")):
            return QuestionType.YES_NO.value

        # 计数类问题
        if re.search(r'\bhow many\b', q_lower):
            return QuestionType.COUNTING.value

        # 多跳判断：多个关系 或 "的...的" 嵌套结构
        if len(relations) >= 2:
            return QuestionType.MULTI_HOP.value

        # 包含 "who/what ... of ... of" 等嵌套
        if len(re.findall(r'\bof\b', q_lower)) >= 2:
            return QuestionType.MULTI_HOP.value

        # 多个实体且有关系连接
        if len(entities) >= 2 and len(relations) >= 1:
            return QuestionType.MULTI_HOP.value

        # 单跳
        if entities or relations:
            return QuestionType.SINGLE_HOP.value

        return QuestionType.UNKNOWN.value

    # ==========================================
    # 问题分解
    # ==========================================

    def _decompose_question(self, question: str, entities: List[str],
                             relations: List[str], query_type: str) -> List[HopQuery]:
        """
        将问题分解为多跳查询

        Args:
            question: 问题文本
            entities: 提取的实体
            relations: 识别的关系
            query_type: 问题类型

        Returns:
            HopQuery 列表
        """
        wh_info = _WH_PATTERNS.get(self._extract_wh_word(question), {})
        default_intent = wh_info.get("intent", "find_entity")
        default_type = wh_info.get("expected_type", "")

        if query_type == QuestionType.SINGLE_HOP.value:
            return self._decompose_single_hop(
                question, entities, relations, default_intent, default_type
            )

        elif query_type in (QuestionType.MULTI_HOP.value, QuestionType.BRIDGE.value):
            return self._decompose_multi_hop(
                question, entities, relations, default_intent, default_type
            )

        elif query_type == QuestionType.COMPARISON.value:
            return self._decompose_comparison(
                question, entities, relations
            )

        else:
            # 默认单跳
            return [HopQuery(
                hop_id=1,
                intent=default_intent,
                target=question,
                expected_type=default_type,
            )]

    def _decompose_single_hop(self, question: str, entities: List[str],
                               relations: List[str], intent: str,
                               expected_type: str) -> List[HopQuery]:
        """单跳问题分解"""
        target = question
        if entities and relations:
            target = f"{relations[0]} of {entities[0]}"
        elif entities:
            target = entities[0]

        return [HopQuery(
            hop_id=1,
            intent=intent,
            target=target,
            expected_type=expected_type,
        )]

    def _decompose_multi_hop(self, question: str, entities: List[str],
                              relations: List[str], intent: str,
                              expected_type: str) -> List[HopQuery]:
        """
        多跳问题分解

        策略：
        - 如果有多个关系，每个关系对应一跳
        - 如果有嵌套 "of" 结构，按 "of" 拆分
        - 最后一跳使用原始问题的疑问词意图
        """
        hops = []

        if len(relations) >= 2:
            # 按关系数量分解
            for i, rel in enumerate(relations):
                entity = entities[i] if i < len(entities) else f"{{result_from_hop_{i}}}"
                hop = HopQuery(
                    hop_id=i + 1,
                    intent="find_entity" if i < len(relations) - 1 else intent,
                    target=f"{rel} of {entity}",
                    expected_type="" if i < len(relations) - 1 else expected_type,
                    depends_on=i if i > 0 else None,
                )
                hops.append(hop)

        elif "of" in question.lower():
            # 按 "of" 嵌套拆分
            parts = self._split_by_of(question)
            for i, part in enumerate(parts):
                hop = HopQuery(
                    hop_id=i + 1,
                    intent="find_entity" if i < len(parts) - 1 else intent,
                    target=part.strip(),
                    expected_type="" if i < len(parts) - 1 else expected_type,
                    depends_on=i if i > 0 else None,
                )
                hops.append(hop)
        else:
            # 无法进一步分解，作为单跳处理
            hops.append(HopQuery(
                hop_id=1,
                intent=intent,
                target=question,
                expected_type=expected_type,
            ))

        return hops if hops else [HopQuery(hop_id=1, intent=intent, target=question)]

    def _decompose_comparison(self, question: str, entities: List[str],
                               relations: List[str]) -> List[HopQuery]:
        """
        比较类问题分解

        为每个被比较的实体创建一个检索跳，最后一跳用于比较。
        """
        hops = []

        if len(entities) >= 2:
            # 为每个实体查询
            for i, entity in enumerate(entities[:2]):
                rel = relations[0] if relations else "attribute"
                hops.append(HopQuery(
                    hop_id=i + 1,
                    intent="find_attribute",
                    target=f"{rel} of {entity}",
                    expected_type="",
                ))
            # 比较跳
            hops.append(HopQuery(
                hop_id=len(hops) + 1,
                intent="compare",
                target=question,
                expected_type="",
                depends_on=None,  # 依赖所有前序跳
            ))
        else:
            hops.append(HopQuery(
                hop_id=1,
                intent="compare",
                target=question,
            ))

        return hops

    # ==========================================
    # 查询生成
    # ==========================================

    def _generate_graph_query(self, entities: List[str], relations: List[str],
                               hops: List[HopQuery]) -> Optional[Dict[str, Any]]:
        """
        生成知识图谱查询（模式匹配格式）

        输出格式：
        {
            "pattern": [("Tesla", "CEO", "?person"), ("?person", "alma_mater", "?school")],
            "filters": [],
            "return": ["?school"]
        }

        Args:
            entities: 提取的实体
            relations: 识别的关系
            hops: 多跳查询

        Returns:
            图谱查询字典，无法生成时返回 None
        """
        if not entities or not relations:
            return None

        pattern = []
        variables = []
        prev_var = entities[0] if entities else "?start"

        for i, (hop, rel) in enumerate(zip(hops, relations)):
            var_name = f"?var_{i + 1}"
            variables.append(var_name)

            # 使用已知实体或变量
            subject = prev_var
            predicate = rel
            obj = var_name

            pattern.append((subject, predicate, obj))
            prev_var = var_name

        if not pattern:
            return None

        return_vars = [variables[-1]] if variables else []

        return {
            "pattern": [list(p) for p in pattern],
            "filters": [],
            "return": return_vars,
        }

    def _generate_vector_query(self, question: str, entities: List[str],
                                relations: List[str]) -> str:
        """
        生成优化的向量检索查询文本

        优化策略：
        - 移除疑问词，保留实质内容
        - 将实体和关系重组为陈述句式

        Args:
            question: 原始问题
            entities: 提取的实体
            relations: 识别的关系

        Returns:
            优化后的查询文本
        """
        # 移除疑问词和标点
        query = question.rstrip("?").strip()

        # 移除开头的疑问词短语
        wh_prefixes = [
            "who is the", "who was the", "who are the",
            "what is the", "what was the", "what are the",
            "where is the", "where was the", "where are the",
            "when was the", "when did the", "when is the",
            "which is the", "which was the",
            "how many", "how much",
            "who is", "who was", "what is", "what was",
            "where is", "where was", "when was", "when did",
            "in which",
        ]

        q_lower = query.lower()
        for prefix in sorted(wh_prefixes, key=len, reverse=True):
            if q_lower.startswith(prefix):
                query = query[len(prefix):].strip()
                break

        # 如果查询变得太短，使用实体+关系拼接
        if len(query) < 5 and (entities or relations):
            parts = entities + relations
            query = " ".join(parts)

        return query if query else question

    # ==========================================
    # 辅助方法
    # ==========================================

    @staticmethod
    def _extract_wh_word(question: str) -> str:
        """提取疑问词"""
        q_lower = question.lower().strip()
        for wh in ["how many", "how much", "how old",
                    "who", "what", "where", "when", "which", "how"]:
            if q_lower.startswith(wh):
                return wh
        return ""

    @staticmethod
    def _split_by_of(question: str) -> List[str]:
        """
        按 "of" 从右向左拆分问题

        "the CEO of the company that made the iPhone"
        -> ["the iPhone", "the company that made", "the CEO"]
        (实际返回反转后：从第一跳到最后一跳)
        """
        # 移除疑问词
        q = question.rstrip("?").strip()
        for prefix in ["who is the", "what is the", "where is the",
                        "who is", "what is", "where is"]:
            if q.lower().startswith(prefix):
                q = q[len(prefix):].strip()
                break

        parts = re.split(r'\s+of\s+', q)

        if len(parts) <= 1:
            return [question]

        # 反转：最内层的实体在前（第一跳），最外层属性在后（最后一跳）
        parts.reverse()
        return parts

    def __repr__(self) -> str:
        return "SemanticParser()"
