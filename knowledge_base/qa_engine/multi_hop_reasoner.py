"""
多跳推理器 (Multi-Hop Reasoner)

处理需要多步推理才能回答的复杂问题。

支持三种推理方案：
1. chain_reasoning: 链式推理 — 逐跳检索 + 结果传递 + 迭代生成下一跳查询
2. graph_reasoning: 图推理 — 将问题转换为图查询模式，在知识图谱上直接执行路径搜索
3. iterative_refinement: 迭代精炼 — 初始检索 → 评估是否足够 → 补充查询 → 重复

每种方案的选择由 QuestionUnderstanding 分析后决定。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data.models import Fact, ReasoningChain, ReasoningStep
from experts.base_expert import BaseExpert
from experts.expert_router import ExpertRouter
from kb_builder.models import SearchResult
from neuro_symbolic.semantic_parser import StructuredQuery
from qa_engine.question_understanding import QuestionAnalysis
from utils.config import Config

logger = logging.getLogger(__name__)


# ==============================================================================
# 推理结果模型
# ==============================================================================


@dataclass
class ReasoningResult:
    """
    推理结果

    Attributes:
        question: 原始问题
        answer: 最终答案文本
        confidence: 置信度 (0.0 ~ 1.0)
        reasoning_chain: 推理链
        retrieved_results: 所有检索到的结果
        strategy: 使用的推理策略
        total_hops: 实际推理跳数
        is_complete: 推理是否完整完成
        time_elapsed: 推理耗时（秒）
        metadata: 附加信息
    """

    question: str = ""
    answer: str = ""
    confidence: float = 0.0
    reasoning_chain: Optional[ReasoningChain] = None
    retrieved_results: List[SearchResult] = field(default_factory=list)
    strategy: str = ""
    total_hops: int = 0
    is_complete: bool = False
    time_elapsed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def supporting_texts(self) -> List[str]:
        """获取所有支撑文本"""
        return [r.text for r in self.retrieved_results if r.text]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning_chain": self.reasoning_chain.to_dict() if self.reasoning_chain else None,
            "retrieved_results": [r.to_dict() for r in self.retrieved_results],
            "strategy": self.strategy,
            "total_hops": self.total_hops,
            "is_complete": self.is_complete,
            "time_elapsed": self.time_elapsed,
        }

    def __repr__(self) -> str:
        ans_preview = self.answer[:40] + "..." if len(self.answer) > 40 else self.answer
        return (
            f"ReasoningResult(answer='{ans_preview}', "
            f"conf={self.confidence:.2f}, hops={self.total_hops}, "
            f"strategy='{self.strategy}')"
        )


# ==============================================================================
# 多跳推理器
# ==============================================================================


class MultiHopReasoner:
    """
    多跳推理器

    根据问题分析结果选择合适的推理策略，协调专家检索和推理过程。

    Usage:
        >>> router = ExpertRouter(experts)
        >>> reasoner = MultiHopReasoner(expert_router=router)
        >>> analysis = QuestionUnderstanding().analyze("Who directed the movie starring Tom Hanks about WWII?")
        >>> result = reasoner.reason(analysis)
        >>> print(result.answer)
    """

    def __init__(
        self,
        expert_router: ExpertRouter,
        config: Optional[Config] = None,
    ):
        """
        初始化多跳推理器

        Args:
            expert_router: 专家路由器
            config: 配置对象
        """
        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self._config = config
        self.expert_router = expert_router
        self.max_hops = config.get("reasoning.max_hops", 5)
        self.max_iterations = config.get("reasoning.max_iterations", 3)
        self.confidence_threshold = config.get("reasoning.confidence_threshold", 0.5)

        logger.info(
            f"MultiHopReasoner 初始化: max_hops={self.max_hops}, "
            f"max_iterations={self.max_iterations}"
        )

    # ==========================================
    # 主接口
    # ==========================================

    def reason(self, analysis: QuestionAnalysis) -> ReasoningResult:
        """
        执行推理

        根据 QuestionAnalysis 中的 reasoning_strategy 选择推理方案。

        Args:
            analysis: 问题分析结果

        Returns:
            推理结果
        """
        start_time = time.time()
        question = analysis.question
        strategy = analysis.reasoning_strategy

        logger.info(
            f"开始推理: strategy='{strategy}', "
            f"hops={analysis.required_hops}, question='{question[:60]}...'"
        )

        try:
            if strategy == "single_retrieval":
                result = self._single_retrieval(analysis)
            elif strategy == "chain_reasoning":
                result = self.chain_reasoning(analysis)
            elif strategy == "graph_reasoning":
                result = self.graph_reasoning(analysis)
            elif strategy == "parallel_retrieval":
                result = self._parallel_retrieval(analysis)
            elif strategy == "iterative_refinement":
                result = self.iterative_refinement(analysis)
            elif strategy == "verification":
                result = self._verification(analysis)
            elif strategy == "aggregate_retrieval":
                result = self._aggregate_retrieval(analysis)
            else:
                logger.warning(f"未知策略 '{strategy}'，回退到单次检索")
                result = self._single_retrieval(analysis)
        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            result = ReasoningResult(
                question=question,
                answer="",
                confidence=0.0,
                strategy=strategy,
                is_complete=False,
            )

        result.time_elapsed = time.time() - start_time
        result.strategy = strategy

        logger.info(
            f"推理完成: answer='{result.answer[:50]}...', "
            f"confidence={result.confidence:.2f}, "
            f"hops={result.total_hops}, time={result.time_elapsed:.2f}s"
        )

        return result

    # ==========================================
    # 方案一：链式推理 (Chain Reasoning)
    # ==========================================

    def chain_reasoning(self, analysis: QuestionAnalysis) -> ReasoningResult:
        """
        链式推理

        流程：
        1. 从第一跳开始：路由到专家 → 检索 → 获得中间结果
        2. 基于中间结果构造下一跳查询
        3. 重复直到得到最终答案或达到最大跳数

        Args:
            analysis: 问题分析结果

        Returns:
            推理结果
        """
        question = analysis.question
        chain = ReasoningChain(question_id=question[:20])
        all_results: List[SearchResult] = []
        context_parts: List[str] = []
        current_query = question

        structured_query = analysis.structured_query
        hop_queries = structured_query.hops if structured_query else []

        for hop_idx in range(min(analysis.required_hops, self.max_hops)):
            hop_num = hop_idx + 1
            logger.debug(f"  Hop {hop_num}: query='{current_query[:50]}...'")

            # 如果有预分解的子查询，使用它
            if hop_idx < len(hop_queries):
                hop_query = hop_queries[hop_idx]
                sub_query = hop_query.target
            else:
                sub_query = current_query

            # 1. 路由到专家
            experts = self.expert_router.route(sub_query, top_k_experts=2)

            # 2. 从各专家检索
            hop_results = self._retrieve_from_experts(
                experts, sub_query, structured_query
            )
            all_results.extend(hop_results)

            # 3. 提取本跳结果
            hop_answer = self._extract_hop_answer(hop_results, sub_query)
            context_parts.append(hop_answer)

            # 4. 记录推理步骤
            step = ReasoningStep(
                hop_id=hop_num,
                query=sub_query,
                result=hop_answer,
                expert_id=experts[0].expert_id if experts else "",
                confidence=hop_results[0].score if hop_results else 0.0,
            )
            chain.add_step(step)

            # 5. 判断是否可以回答
            if self._can_answer(question, context_parts, hop_results):
                break

            # 6. 生成下一跳查询
            current_query = self._generate_next_query(
                question, context_parts, hop_answer
            )

        # 从所有上下文中提取最终答案
        final_answer = self._extract_final_answer(question, context_parts, all_results)
        confidence = self._compute_confidence(chain, all_results)

        chain.final_answer = final_answer
        chain.is_complete = True

        return ReasoningResult(
            question=question,
            answer=final_answer,
            confidence=confidence,
            reasoning_chain=chain,
            retrieved_results=all_results,
            total_hops=chain.total_hops,
            is_complete=True,
        )

    # ==========================================
    # 方案二：图推理 (Graph Reasoning)
    # ==========================================

    def graph_reasoning(self, analysis: QuestionAnalysis) -> ReasoningResult:
        """
        图推理

        利用结构化查询在知识图谱上直接执行路径搜索。

        流程：
        1. 使用 SemanticParser 生成的 graph_query
        2. 在各专家的知识图谱上执行模式匹配
        3. 收集所有匹配路径
        4. 如果图谱查询无结果，回退到链式推理

        Args:
            analysis: 问题分析结果

        Returns:
            推理结果
        """
        question = analysis.question
        structured_query = analysis.structured_query
        graph_query = structured_query.graph_query if structured_query else None

        if not graph_query:
            logger.info("无图谱查询，回退到链式推理")
            return self.chain_reasoning(analysis)

        chain = ReasoningChain(question_id=question[:20])
        all_results: List[SearchResult] = []

        # 路由到相关专家
        experts = self.expert_router.route(question, top_k_experts=3)

        # 在各专家的图谱上执行查询
        for expert in experts:
            if not expert.has_knowledge_graph:
                continue

            # 执行模式匹配
            symbolic_query = {"pattern": [tuple(p) for p in graph_query.get("pattern", [])]}
            results = expert.retrieve_symbolic(symbolic_query)
            all_results.extend(results)

        if all_results:
            # 从图谱结果中提取答案
            final_answer = self._extract_answer_from_graph_results(all_results, graph_query)
            confidence = min(1.0, max(r.score for r in all_results))

            # 记录推理步骤
            step = ReasoningStep(
                hop_id=1,
                query=str(graph_query.get("pattern", [])),
                result=final_answer,
                confidence=confidence,
            )
            chain.add_step(step)
            chain.final_answer = final_answer
            chain.is_complete = True

            return ReasoningResult(
                question=question,
                answer=final_answer,
                confidence=confidence,
                reasoning_chain=chain,
                retrieved_results=all_results,
                total_hops=1,
                is_complete=True,
                metadata={"graph_query": graph_query},
            )

        # 图谱无结果，回退到链式推理
        logger.info("图谱查询无结果，回退到链式推理")
        return self.chain_reasoning(analysis)

    # ==========================================
    # 方案三：迭代精炼 (Iterative Refinement)
    # ==========================================

    def iterative_refinement(self, analysis: QuestionAnalysis) -> ReasoningResult:
        """
        迭代精炼推理

        流程：
        1. 初始检索（广泛收集信息）
        2. 评估当前信息是否足以回答
        3. 如果不够，分析缺什么信息，生成补充查询
        4. 重复直到有足够信息或达到最大迭代次数

        Args:
            analysis: 问题分析结果

        Returns:
            推理结果
        """
        question = analysis.question
        chain = ReasoningChain(question_id=question[:20])
        all_results: List[SearchResult] = []
        context_parts: List[str] = []

        for iteration in range(self.max_iterations):
            iter_num = iteration + 1
            logger.debug(f"  Iteration {iter_num}/{self.max_iterations}")

            # 确定本轮查询
            if iteration == 0:
                # 第一轮：使用原始问题 + 优化后的向量查询
                current_query = (
                    analysis.structured_query.vector_query
                    if analysis.structured_query else question
                )
            else:
                # 后续轮：根据已有上下文生成补充查询
                current_query = self._generate_refinement_query(
                    question, context_parts
                )

            # 路由到专家
            experts = self.expert_router.route(current_query, top_k_experts=2)

            # 检索
            results = self._retrieve_from_experts(
                experts, current_query, analysis.structured_query
            )
            all_results.extend(results)

            # 更新上下文
            for r in results[:3]:  # 取 top 3 结果
                if r.text and r.text not in context_parts:
                    context_parts.append(r.text)

            # 记录步骤
            hop_answer = self._extract_hop_answer(results, current_query)
            step = ReasoningStep(
                hop_id=iter_num,
                query=current_query,
                result=hop_answer,
                expert_id=experts[0].expert_id if experts else "",
                confidence=results[0].score if results else 0.0,
            )
            chain.add_step(step)

            # 评估是否足够回答
            if self._can_answer(question, context_parts, all_results):
                break

        # 生成最终答案
        final_answer = self._extract_final_answer(question, context_parts, all_results)
        confidence = self._compute_confidence(chain, all_results)

        chain.final_answer = final_answer
        chain.is_complete = True

        return ReasoningResult(
            question=question,
            answer=final_answer,
            confidence=confidence,
            reasoning_chain=chain,
            retrieved_results=all_results,
            total_hops=chain.total_hops,
            is_complete=True,
        )

    # ==========================================
    # 辅助推理策略
    # ==========================================

    def _single_retrieval(self, analysis: QuestionAnalysis) -> ReasoningResult:
        """单次检索（单跳问题）"""
        question = analysis.question
        chain = ReasoningChain(question_id=question[:20])

        # 构造查询
        vector_query = question
        if analysis.structured_query and analysis.structured_query.vector_query:
            vector_query = analysis.structured_query.vector_query

        # 路由 + 检索
        experts = self.expert_router.route(vector_query, top_k_experts=2)
        results = self._retrieve_from_experts(
            experts, vector_query, analysis.structured_query
        )

        # 提取答案
        answer = self._extract_hop_answer(results, question)
        confidence = results[0].score if results else 0.0

        step = ReasoningStep(
            hop_id=1, query=vector_query, result=answer,
            expert_id=experts[0].expert_id if experts else "",
            confidence=confidence,
        )
        chain.add_step(step)
        chain.final_answer = answer
        chain.is_complete = True

        return ReasoningResult(
            question=question,
            answer=answer,
            confidence=confidence,
            reasoning_chain=chain,
            retrieved_results=results,
            total_hops=1,
            is_complete=True,
        )

    def _parallel_retrieval(self, analysis: QuestionAnalysis) -> ReasoningResult:
        """并行检索（比较类问题）"""
        question = analysis.question
        chain = ReasoningChain(question_id=question[:20])
        all_results: List[SearchResult] = []
        entity_results: Dict[str, List[SearchResult]] = {}

        # 为每个关键实体分别检索
        entities = analysis.key_entities
        if len(entities) < 2:
            # 无法比较，回退到链式
            return self.chain_reasoning(analysis)

        for i, entity in enumerate(entities[:2]):
            sub_query = f"{entity} {' '.join(analysis.relations)}" if analysis.relations else entity
            experts = self.expert_router.route(sub_query, top_k_experts=2)
            results = self._retrieve_from_experts(experts, sub_query, analysis.structured_query)
            entity_results[entity] = results
            all_results.extend(results)

            step = ReasoningStep(
                hop_id=i + 1,
                query=sub_query,
                result=self._extract_hop_answer(results, sub_query),
                expert_id=experts[0].expert_id if experts else "",
                confidence=results[0].score if results else 0.0,
            )
            chain.add_step(step)

        # 比较结果
        context_parts = [
            self._extract_hop_answer(entity_results.get(e, []), e)
            for e in entities[:2]
        ]
        final_answer = self._extract_final_answer(question, context_parts, all_results)
        confidence = self._compute_confidence(chain, all_results)

        chain.final_answer = final_answer
        chain.is_complete = True

        return ReasoningResult(
            question=question,
            answer=final_answer,
            confidence=confidence,
            reasoning_chain=chain,
            retrieved_results=all_results,
            total_hops=chain.total_hops,
            is_complete=True,
        )

    def _verification(self, analysis: QuestionAnalysis) -> ReasoningResult:
        """验证型推理（Yes/No 问题）"""
        result = self._single_retrieval(analysis)
        # 对于 Yes/No 问题，检查检索结果是否支持
        if result.retrieved_results:
            # 有强证据 → Yes
            if result.confidence >= self.confidence_threshold:
                result.answer = "Yes"
            else:
                result.answer = "No"
        return result

    def _aggregate_retrieval(self, analysis: QuestionAnalysis) -> ReasoningResult:
        """聚合型推理（计数类问题）"""
        result = self._single_retrieval(analysis)
        # 对于计数问题，尝试从结果中统计数量
        return result

    # ==========================================
    # 核心辅助方法
    # ==========================================

    def _retrieve_from_experts(
        self,
        experts: List[BaseExpert],
        query: str,
        structured_query: Optional[StructuredQuery] = None,
    ) -> List[SearchResult]:
        """
        从多个专家处检索

        同时进行向量检索和符号检索（如果有结构化查询）。

        Args:
            experts: 专家列表
            query: 查询文本
            structured_query: 结构化查询（用于符号检索）

        Returns:
            合并后的检索结果
        """
        all_results = []
        top_k = self._config.get("retrieval.vector_top_k", 5)
        alpha = self._config.get("retrieval.hybrid_alpha", 0.7)

        # 构造符号查询
        symbolic_query = None
        if structured_query and structured_query.graph_query:
            pattern = structured_query.graph_query.get("pattern", [])
            if pattern:
                symbolic_query = {"pattern": [tuple(p) for p in pattern]}

        for expert in experts:
            try:
                results = expert.hybrid_retrieve(
                    query=query,
                    symbolic_query=symbolic_query,
                    top_k=top_k,
                    alpha=alpha,
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"从专家 {expert.expert_id} 检索失败: {e}")

        # 按分数排序并去重
        all_results = self._deduplicate_results(all_results)
        all_results.sort(key=lambda r: r.score, reverse=True)

        return all_results

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """去重检索结果"""
        seen_texts = set()
        unique = []
        for r in results:
            text_key = r.text[:100] if r.text else r.id
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique.append(r)
        return unique

    def _extract_hop_answer(self, results: List[SearchResult],
                             query: str) -> str:
        """
        从一跳的检索结果中提取答案

        优先使用符号结果（精确），否则使用向量结果（语义）。

        Args:
            results: 检索结果
            query: 本跳查询

        Returns:
            答案文本
        """
        if not results:
            return ""

        # 优先使用符号结果
        symbolic = [r for r in results if r.source in ("symbolic", "graph")]
        if symbolic:
            best = symbolic[0]
            # 如果是模式匹配结果，提取绑定值
            bindings = best.metadata.get("bindings", {})
            if bindings:
                values = list(bindings.values())
                return values[-1] if values else best.text
            entity = best.metadata.get("entity", "")
            if entity:
                return entity
            return best.text

        # 使用向量检索结果
        return results[0].text

    def _extract_final_answer(self, question: str, context_parts: List[str],
                               results: List[SearchResult]) -> str:
        """
        从所有上下文和检索结果中提取最终答案

        Args:
            question: 原始问题
            context_parts: 各跳收集到的上下文
            results: 所有检索结果

        Returns:
            最终答案文本
        """
        # 策略 1: 如果最后一个 context_part 是从符号系统得到的精确实体
        if context_parts:
            last_ctx = context_parts[-1]
            # 短答案（实体名称等）直接返回
            if len(last_ctx.split()) <= 10:
                return last_ctx

        # 策略 2: 从所有结果中找到分数最高的
        if results:
            # 优先符号结果
            symbolic = [r for r in results if r.source in ("symbolic", "graph")]
            if symbolic:
                bindings = symbolic[0].metadata.get("bindings", {})
                if bindings:
                    return list(bindings.values())[-1]
                entity = symbolic[0].metadata.get("entity", "")
                if entity:
                    return entity

            # 向量结果
            return results[0].text

        # 无结果
        if context_parts:
            return context_parts[-1]

        return ""

    def _extract_answer_from_graph_results(
        self,
        results: List[SearchResult],
        graph_query: Dict[str, Any],
    ) -> str:
        """从图谱查询结果中提取答案"""
        return_vars = graph_query.get("return", [])

        for r in results:
            bindings = r.metadata.get("bindings", {})
            if bindings:
                # 返回指定的变量值
                for var in return_vars:
                    if var in bindings:
                        return bindings[var]
                # 没有指定返回变量，返回最后一个绑定值
                return list(bindings.values())[-1]

            entity = r.metadata.get("entity", "")
            if entity:
                return entity

        return results[0].text if results else ""

    def _can_answer(self, question: str, context_parts: List[str],
                     results: List[SearchResult]) -> bool:
        """
        判断当前信息是否足以回答问题

        简单启发式：
        - 如果有高置信度的符号结果 → 可以回答
        - 如果上下文中的信息覆盖了问题中的关键实体 → 可以回答

        Args:
            question: 原始问题
            context_parts: 已收集的上下文
            results: 所有检索结果

        Returns:
            是否可以回答
        """
        if not results:
            return False

        # 高置信度符号结果
        symbolic = [r for r in results if r.source in ("symbolic", "graph")]
        if symbolic and symbolic[0].score >= 0.9:
            return True

        # 至少有 3 条相关的检索结果
        high_quality = [r for r in results if r.score >= self.confidence_threshold]
        if len(high_quality) >= 3:
            return True

        return False

    def _generate_next_query(self, original_question: str,
                              context_parts: List[str],
                              last_answer: str) -> str:
        """
        基于当前上下文生成下一跳查询

        简单策略：用上一跳的答案替换原始问题中的占位部分。

        Args:
            original_question: 原始问题
            context_parts: 已有上下文
            last_answer: 上一跳的答案

        Returns:
            下一跳查询文本
        """
        if not last_answer:
            return original_question

        # 策略：将上一跳答案与原始问题结合
        # 例如：原问题 "Who is the CEO of the company that made iPhone?"
        # 上一跳得到 "Apple" → 下一跳 "Who is the CEO of Apple?"
        combined = f"{original_question} Context: {last_answer}"

        # 简化：如果答案是一个实体名称（短文本），用它替换
        if len(last_answer.split()) <= 5:
            return f"{last_answer} {original_question}"

        return combined

    def _generate_refinement_query(self, question: str,
                                    context_parts: List[str]) -> str:
        """
        生成迭代精炼的补充查询

        分析已有上下文缺少什么信息。

        Args:
            question: 原始问题
            context_parts: 已有上下文

        Returns:
            补充查询文本
        """
        # 简单策略：用问题中未在上下文出现的关键词重新查询
        context_text = " ".join(context_parts).lower()
        question_words = set(question.lower().split())
        missing_words = [
            w for w in question_words
            if w not in context_text and len(w) > 3
            and w not in {"what", "where", "when", "which", "that", "this", "with", "from"}
        ]

        if missing_words:
            return " ".join(missing_words) + " " + question
        return question

    def _compute_confidence(self, chain: ReasoningChain,
                             results: List[SearchResult]) -> float:
        """
        计算整体推理置信度

        综合各跳的置信度：取平均值，并根据推理完整性调整。

        Args:
            chain: 推理链
            results: 所有检索结果

        Returns:
            置信度 (0.0 ~ 1.0)
        """
        if not chain.steps:
            return 0.0

        # 各跳置信度的加权平均
        step_confs = [s.confidence for s in chain.steps if s.confidence > 0]
        if step_confs:
            avg_conf = sum(step_confs) / len(step_confs)
        else:
            avg_conf = 0.0

        # 根据检索结果质量调整
        if results:
            top_score = max(r.score for r in results)
            avg_conf = 0.6 * avg_conf + 0.4 * top_score

        return min(max(avg_conf, 0.0), 1.0)

    def __repr__(self) -> str:
        return (
            f"MultiHopReasoner(max_hops={self.max_hops}, "
            f"max_iterations={self.max_iterations})"
        )
