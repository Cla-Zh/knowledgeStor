"""
问题理解模块 (Question Understanding)

对用户问题进行深度分析，为后续的推理和检索提供决策依据。

分析维度：
- question_type: 问题类型（单跳/多跳/比较/计数/是非）
- complexity: 复杂度评分（1-10）
- required_hops: 预估推理跳数
- key_entities: 关键实体
- relations: 涉及的关系
- structured_query: 结构化查询（来自 SemanticParser）

协调 SemanticParser 完成语义解析，并在此基础上补充额外的分析信息。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data.models import QuestionType
from neuro_symbolic.semantic_parser import SemanticParser, StructuredQuery
from utils.config import Config

logger = logging.getLogger(__name__)


# ==============================================================================
# 分析结果模型
# ==============================================================================


@dataclass
class QuestionAnalysis:
    """
    问题分析结果

    Attributes:
        question: 原始问题文本
        question_type: 问题类型
        complexity: 复杂度评分 (1-10)
        required_hops: 预估需要的推理跳数
        key_entities: 关键实体列表
        relations: 涉及的关系列表
        structured_query: 结构化查询（来自 SemanticParser）
        reasoning_strategy: 建议的推理策略
        metadata: 附加分析信息
    """

    question: str = ""
    question_type: str = QuestionType.UNKNOWN.value
    complexity: float = 1.0
    required_hops: int = 1
    key_entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    structured_query: Optional[StructuredQuery] = None
    reasoning_strategy: str = "single_retrieval"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi_hop(self) -> bool:
        return self.required_hops > 1

    @property
    def is_comparison(self) -> bool:
        return self.question_type == QuestionType.COMPARISON.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "question_type": self.question_type,
            "complexity": self.complexity,
            "required_hops": self.required_hops,
            "key_entities": self.key_entities,
            "relations": self.relations,
            "structured_query": self.structured_query.to_dict() if self.structured_query else None,
            "reasoning_strategy": self.reasoning_strategy,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"QuestionAnalysis(type='{self.question_type}', "
            f"complexity={self.complexity:.1f}, hops={self.required_hops}, "
            f"strategy='{self.reasoning_strategy}')"
        )


# ==============================================================================
# 问题理解模块
# ==============================================================================


# 推理策略映射
_STRATEGY_MAP = {
    QuestionType.SINGLE_HOP.value: "single_retrieval",
    QuestionType.MULTI_HOP.value: "chain_reasoning",
    QuestionType.BRIDGE.value: "chain_reasoning",
    QuestionType.COMPARISON.value: "parallel_retrieval",
    QuestionType.COUNTING.value: "aggregate_retrieval",
    QuestionType.YES_NO.value: "verification",
}


class QuestionUnderstanding:
    """
    问题理解模块

    分析用户问题的类型、复杂度、关键信息，为推理引擎提供决策依据。

    Usage:
        >>> qu = QuestionUnderstanding()
        >>> analysis = qu.analyze("Who is the CEO of the company that made the iPhone?")
        >>> print(analysis.question_type)      # "multi_hop"
        >>> print(analysis.required_hops)      # 2
        >>> print(analysis.reasoning_strategy) # "chain_reasoning"
        >>> print(analysis.key_entities)       # ["iPhone"]
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化问题理解模块

        Args:
            config: 配置对象
        """
        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self._config = config
        self._parser = SemanticParser(config=config)

        logger.info("QuestionUnderstanding 初始化完成")

    # ==========================================
    # 主接口
    # ==========================================

    def analyze(self, question: str) -> QuestionAnalysis:
        """
        分析问题

        完整流程：
        1. 调用 SemanticParser 获取结构化查询
        2. 计算复杂度
        3. 估算推理跳数
        4. 选择推理策略
        5. 汇总分析结果

        Args:
            question: 自然语言问题

        Returns:
            QuestionAnalysis 分析结果
        """
        question = question.strip()
        if not question:
            return QuestionAnalysis(question=question)

        logger.debug(f"分析问题: {question}")

        # Step 1: 语义解析
        structured_query = self._parser.parse(question)

        # Step 2: 计算复杂度
        complexity = self._compute_complexity(question, structured_query)

        # Step 3: 估算推理跳数
        required_hops = self._estimate_hops(structured_query, complexity)

        # Step 4: 确定问题类型（综合语义解析和额外分析）
        question_type = self._refine_question_type(
            structured_query.query_type, required_hops, question
        )

        # Step 5: 选择推理策略
        reasoning_strategy = self._select_strategy(question_type, required_hops, complexity)

        analysis = QuestionAnalysis(
            question=question,
            question_type=question_type,
            complexity=complexity,
            required_hops=required_hops,
            key_entities=structured_query.entities,
            relations=structured_query.relations,
            structured_query=structured_query,
            reasoning_strategy=reasoning_strategy,
            metadata={
                "wh_word": structured_query.metadata.get("wh_word", ""),
                "num_hops_from_parser": structured_query.num_hops,
                "has_graph_query": structured_query.graph_query is not None,
            },
        )

        logger.debug(f"分析结果: {analysis}")
        return analysis

    # ==========================================
    # 复杂度计算
    # ==========================================

    def _compute_complexity(self, question: str,
                             structured_query: StructuredQuery) -> float:
        """
        计算问题复杂度 (1.0 ~ 10.0)

        评分因素：
        - 问题长度
        - 实体数量
        - 关系数量
        - 跳数
        - 特殊结构（比较、嵌套等）

        Args:
            question: 问题文本
            structured_query: 结构化查询

        Returns:
            复杂度评分
        """
        score = 1.0

        # 因素 1: 问题长度 (0 ~ 2.0)
        word_count = len(question.split())
        score += min(word_count / 15.0, 2.0)

        # 因素 2: 实体数量 (0 ~ 2.0)
        n_entities = len(structured_query.entities)
        score += min(n_entities * 0.5, 2.0)

        # 因素 3: 关系数量 (0 ~ 2.0)
        n_relations = len(structured_query.relations)
        score += min(n_relations * 0.7, 2.0)

        # 因素 4: 解析出的跳数 (0 ~ 2.0)
        n_hops = structured_query.num_hops
        score += min((n_hops - 1) * 1.0, 2.0)

        # 因素 5: 特殊结构 (0 ~ 2.0)
        q_lower = question.lower()
        if structured_query.query_type == QuestionType.COMPARISON.value:
            score += 1.5
        if "and" in q_lower and ("both" in q_lower or "respectively" in q_lower):
            score += 1.0
        if re.search(r'\b(if|suppose|assuming)\b', q_lower):
            score += 0.5

        return min(max(score, 1.0), 10.0)

    # ==========================================
    # 跳数估算
    # ==========================================

    def _estimate_hops(self, structured_query: StructuredQuery,
                        complexity: float) -> int:
        """
        估算推理所需的跳数

        综合考虑：
        - SemanticParser 分解出的跳数
        - 问题复杂度
        - 实体和关系的数量

        Args:
            structured_query: 结构化查询
            complexity: 复杂度评分

        Returns:
            估算的跳数
        """
        # 基础跳数：来自语义解析器
        parser_hops = structured_query.num_hops

        # 基于复杂度的补充判断
        if complexity >= 7.0 and parser_hops <= 1:
            # 高复杂度但解析器只给出 1 跳，可能需要更多
            estimated = max(2, len(structured_query.relations))
        elif complexity >= 4.0 and parser_hops <= 1:
            estimated = max(1, len(structured_query.relations))
        else:
            estimated = parser_hops

        # 比较类问题至少 2 跳
        if structured_query.query_type == QuestionType.COMPARISON.value:
            estimated = max(estimated, 2)

        # 上限
        max_hops = self._config.get("reasoning.max_hops", 5)
        return min(max(estimated, 1), max_hops)

    # ==========================================
    # 问题类型精炼
    # ==========================================

    def _refine_question_type(self, parser_type: str, required_hops: int,
                               question: str) -> str:
        """
        在语义解析器结果的基础上精炼问题类型

        Args:
            parser_type: 解析器给出的类型
            required_hops: 估算的跳数
            question: 原始问题

        Returns:
            精炼后的问题类型
        """
        # 如果解析器给出了明确类型，通常保留
        if parser_type != QuestionType.UNKNOWN.value:
            # 但如果跳数 > 1 且类型为 single_hop，升级为 multi_hop
            if parser_type == QuestionType.SINGLE_HOP.value and required_hops > 1:
                return QuestionType.MULTI_HOP.value
            return parser_type

        # 解析器未知时，根据跳数推断
        if required_hops > 1:
            return QuestionType.MULTI_HOP.value

        return QuestionType.SINGLE_HOP.value

    # ==========================================
    # 推理策略选择
    # ==========================================

    def _select_strategy(self, question_type: str, required_hops: int,
                          complexity: float) -> str:
        """
        选择推理策略

        策略类型：
        - single_retrieval: 单次检索即可回答
        - chain_reasoning: 链式多跳推理
        - parallel_retrieval: 并行检索后比较
        - aggregate_retrieval: 聚合检索后统计
        - verification: 检索验证型
        - iterative_refinement: 迭代精炼型

        Args:
            question_type: 问题类型
            required_hops: 跳数
            complexity: 复杂度

        Returns:
            策略名称
        """
        # 高复杂度使用迭代精炼
        reasoning_mode = self._config.get("reasoning.reasoning_mode", "chain")
        if complexity >= 8.0 and reasoning_mode == "iterative":
            return "iterative_refinement"

        # 从映射表获取默认策略
        strategy = _STRATEGY_MAP.get(question_type, "single_retrieval")

        # 如果是多跳但配置了图推理模式
        if strategy == "chain_reasoning" and reasoning_mode == "graph":
            strategy = "graph_reasoning"

        return strategy

    # ==========================================
    # 批量分析
    # ==========================================

    def analyze_batch(self, questions: List[str]) -> List[QuestionAnalysis]:
        """
        批量分析问题

        Args:
            questions: 问题文本列表

        Returns:
            分析结果列表
        """
        results = []
        for q in questions:
            try:
                analysis = self.analyze(q)
                results.append(analysis)
            except Exception as e:
                logger.warning(f"分析问题失败: '{q[:50]}...': {e}")
                results.append(QuestionAnalysis(question=q))
        return results

    def __repr__(self) -> str:
        return "QuestionUnderstanding()"
