"""
答案生成器 (Answer Generator)

基于检索结果和推理路径生成最终的自然语言答案。

生成策略：
1. 直接返回：如果有明确的符号推理结果（实体名称等），直接返回
2. 抽取式：从检索到的上下文中抽取答案片段
3. 生成式：通过 OpenAI 兼容 Chat API 综合多个来源生成答案

所有模型调用均通过远程 API 完成，无需本地 GPU 或模型文件。

输出包含：
- 答案文本
- 置信度
- 支撑事实
- 推理路径解释
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data.models import Fact, ReasoningChain, ReasoningStep
from kb_builder.models import SearchResult
from qa_engine.multi_hop_reasoner import ReasoningResult
from qa_engine.question_understanding import QuestionAnalysis
from utils.config import Config

logger = logging.getLogger(__name__)


# ==============================================================================
# 答案模型
# ==============================================================================


@dataclass
class Answer:
    """
    最终答案

    Attributes:
        text: 答案文本
        confidence: 置信度 (0.0 ~ 1.0)
        supporting_facts: 支撑事实列表
        reasoning_path: 推理路径（人类可读的逐步解释）
        explanation: 推理过程的自然语言解释
        source: 答案来源 ('symbolic', 'extractive', 'generative')
        metadata: 附加信息
    """

    text: str = ""
    confidence: float = 0.0
    supporting_facts: List[str] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    explanation: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "supporting_facts": self.supporting_facts,
            "reasoning_path": self.reasoning_path,
            "explanation": self.explanation,
            "source": self.source,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"Answer(text='{text_preview}', confidence={self.confidence:.2f}, "
            f"source='{self.source}')"
        )


# ==============================================================================
# 答案生成器
# ==============================================================================


class AnswerGenerator:
    """
    答案生成器

    根据推理结果生成结构化的最终答案，包含支撑证据和推理路径。
    生成式答案通过 OpenAI 兼容的 Chat API 完成。

    Usage:
        >>> generator = AnswerGenerator()
        >>> answer = generator.generate(analysis, reasoning_result)
        >>> print(answer.text)
        >>> print(answer.explanation)
        >>> print(answer.confidence)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化答案生成器

        Args:
            config: 配置对象
        """
        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self._config = config

        # 从配置读取 Chat API 参数
        self.api_key = config.get("api.chat.api_key", "sk-xxx")
        self.base_url = config.get("api.chat.base_url", "https://api.openai.com/v1")
        self.model = config.get("api.chat.model", "gpt-4o-mini")
        self.temperature = config.get("api.chat.temperature", 0.0)
        self.max_tokens = config.get("api.chat.max_tokens", 512)
        self.max_retries = config.get("api.chat.max_retries", 3)
        self.timeout = config.get("api.chat.timeout", 60)

        # OpenAI 客户端（延迟初始化）
        self._client = None

        logger.info(f"AnswerGenerator 初始化完成: model={self.model}")

    # ==========================================
    # API 客户端
    # ==========================================

    def _get_client(self):
        """
        获取 OpenAI Chat 客户端（延迟初始化）

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
            logger.info(f"OpenAI Chat 客户端已初始化: {self.base_url}")

        return self._client

    # ==========================================
    # 主接口
    # ==========================================

    def generate(
        self,
        analysis: QuestionAnalysis,
        reasoning_result: ReasoningResult,
    ) -> Answer:
        """
        生成最终答案

        决策流程（优先级从高到低）：
        1. 如果推理结果有符号推理得到的精确答案 → 直接返回
        2. 调用 Chat API → 生成式回答（简短精准）
        3. 如果有检索结果 → 抽取式回答
        4. 回退：返回最佳检索结果

        Args:
            analysis: 问题分析结果
            reasoning_result: 推理结果

        Returns:
            Answer 答案对象
        """
        question = analysis.question
        logger.debug(f"生成答案: question='{question[:50]}...'")

        # 收集支撑信息
        supporting_facts = self._collect_supporting_facts(reasoning_result)
        reasoning_path = self._build_reasoning_path(reasoning_result)

        # 策略 1: 符号结果直接返回
        answer = self._try_symbolic_answer(reasoning_result)
        if answer:
            answer.supporting_facts = supporting_facts
            answer.reasoning_path = reasoning_path
            answer.explanation = self._generate_explanation(
                question, answer.text, reasoning_path
            )
            logger.debug(f"  符号答案: {answer.text}")
            return answer

        # 策略 2: 生成式答案（优先级提高，确保答案简短精准）
        answer = self._try_generative_answer(question, reasoning_result)
        if answer:
            answer.supporting_facts = supporting_facts
            answer.reasoning_path = reasoning_path
            answer.explanation = self._generate_explanation(
                question, answer.text, reasoning_path
            )
            logger.debug(f"  生成式答案: {answer.text[:50]}...")
            return answer

        # 策略 3: 抽取式答案（作为后备）
        answer = self._try_extractive_answer(question, reasoning_result)
        if answer:
            answer.supporting_facts = supporting_facts
            answer.reasoning_path = reasoning_path
            answer.explanation = self._generate_explanation(
                question, answer.text, reasoning_path
            )
            logger.debug(f"  抽取式答案: {answer.text[:50]}...")
            return answer

        # 回退：返回简短的错误信息
        return Answer(
            text="无法确定",
            confidence=reasoning_result.confidence * 0.3,
            supporting_facts=supporting_facts,
            reasoning_path=reasoning_path,
            explanation=self._generate_explanation(
                question, "无法确定", reasoning_path
            ),
            source="fallback",
        )

    # ==========================================
    # 策略 1: 符号答案
    # ==========================================

    def _try_symbolic_answer(self, reasoning_result: ReasoningResult) -> Optional[Answer]:
        """
        尝试从符号推理结果中提取答案

        如果推理链的最终结果是精确的实体/属性值，直接使用。

        Args:
            reasoning_result: 推理结果

        Returns:
            Answer 或 None
        """
        # 检查推理链是否有精确答案
        chain = reasoning_result.reasoning_chain
        if chain and chain.final_answer:
            answer_text = chain.final_answer
            # 短答案（实体名称等）且是来自符号检索的，高置信度
            if len(answer_text.split()) <= 10:
                # 验证是否来自符号系统
                symbolic_results = [
                    r for r in reasoning_result.retrieved_results
                    if r.source in ("symbolic", "graph", "hybrid")
                ]
                if symbolic_results:
                    return Answer(
                        text=answer_text,
                        confidence=min(reasoning_result.confidence * 1.1, 1.0),
                        source="symbolic",
                    )

        # 从符号检索结果中直接提取
        for result in reasoning_result.retrieved_results:
            if result.source in ("symbolic", "graph"):
                bindings = result.metadata.get("bindings", {})
                if bindings:
                    answer_text = list(bindings.values())[-1]
                    return Answer(
                        text=answer_text,
                        confidence=result.score,
                        source="symbolic",
                        metadata={"bindings": bindings},
                    )
                entity = result.metadata.get("entity", "")
                if entity:
                    return Answer(
                        text=entity,
                        confidence=result.score,
                        source="symbolic",
                    )

        return None

    # ==========================================
    # 策略 2: 抽取式答案
    # ==========================================

    def _try_extractive_answer(self, question: str,
                                reasoning_result: ReasoningResult) -> Optional[Answer]:
        """
        从检索结果中抽取答案

        策略：
        1. 从最相关的文档中查找与问题匹配的答案片段
        2. 如果推理链有结果，使用最后一步的结果

        Args:
            question: 问题文本
            reasoning_result: 推理结果

        Returns:
            Answer 或 None
        """
        # 如果推理链有答案
        if reasoning_result.answer:
            return Answer(
                text=reasoning_result.answer,
                confidence=reasoning_result.confidence,
                source="extractive",
            )

        # 从检索结果中抽取
        results = reasoning_result.retrieved_results
        if not results:
            return None

        # 取最高分的结果
        best_result = results[0]
        answer_text = self._extract_answer_span(question, best_result.text)

        if answer_text:
            return Answer(
                text=answer_text,
                confidence=best_result.score * 0.9,
                source="extractive",
            )

        return None

    def _extract_answer_span(self, question: str, context: str) -> str:
        """
        从上下文中抽取答案片段

        简单启发式方法：
        - 查找与问题关键词邻近的名词短语
        - 匹配常见的答案模式（"is X", "was X", "in X" 等）

        Args:
            question: 问题
            context: 上下文

        Returns:
            答案片段
        """
        if not context:
            return ""

        # 如果上下文本身很短（可能已经是答案），直接返回
        if len(context.split()) <= 15:
            return context

        q_lower = question.lower()

        # 尝试匹配 "X is/was Y" 模式
        # 从问题中提取主语
        for entity_pattern in [
            r'(?:who|what) (?:is|was|are|were) (?:the )?(.+?)(?:\?|$)',
            r'(?:where) (?:is|was|are|were) (.+?)(?:\?|$)',
        ]:
            match = re.search(entity_pattern, q_lower)
            if match:
                subject = match.group(1).strip()
                # 在上下文中找 "subject is/was X"
                answer_pattern = re.compile(
                    re.escape(subject) + r'\s+(?:is|was|are|were)\s+(.+?)(?:\.|,|;|$)',
                    re.IGNORECASE
                )
                ans_match = answer_pattern.search(context)
                if ans_match:
                    return ans_match.group(1).strip()

        # 返回第一个句子作为回退
        sentences = re.split(r'(?<=[.!?])\s+', context)
        if sentences:
            return sentences[0]

        return context[:200]

    # ==========================================
    # 策略 3: 生成式答案（Chat API）
    # ==========================================

    def _try_generative_answer(self, question: str,
                                reasoning_result: ReasoningResult) -> Optional[Answer]:
        """
        通过 OpenAI 兼容 Chat API 生成简短精准的答案

        Args:
            question: 问题
            reasoning_result: 推理结果

        Returns:
            Answer 或 None
        """
        if not reasoning_result.retrieved_results:
            return None

        # 构建上下文（取前 5 个最相关的检索结果）
        context_parts = []
        for r in reasoning_result.retrieved_results[:5]:
            if r.text:
                context_parts.append(r.text)
        context = "\n\n".join(context_parts)

        if not context:
            return None

        # 通过 Chat API 生成简短精准答案
        generated = self._generate_with_api(question, context)
        if generated:
            return Answer(
                text=generated,
                confidence=reasoning_result.confidence * 0.85,
                source="generative",
            )

        return None

    def _generate_with_api(self, question: str, context: str) -> str:
        """
        调用 OpenAI Chat API 生成简短精准的答案

        强调：答案必须简短精准（如数值、实体名称、短语等），不要冗长解释。

        Args:
            question: 问题
            context: 上下文

        Returns:
            生成的答案文本，失败返回空字符串
        """
        try:
            client = self._get_client()

            # 针对中文军事多跳问答优化的 prompt
            system_prompt = (
                "你是一个精准的问答助手。请**严格**根据提供的文档回答问题。\n"
                "要求：\n"
                "1. 答案必须**简短精准**：只返回答案本身（数字、名称、短语等），不要额外解释\n"
                "2. 如果答案是数值，直接返回数值和单位（如 \"8735米\"、\"20架\"）\n"
                "3. 如果答案是名称，只返回名称本身（如 \"艾森豪威尔号\"、\"杰森·黑格\"）\n"
                "4. 如果答案是日期，使用简短格式（如 \"2022年\"、\"4月1日\"）\n"
                "5. **严禁**返回完整段落或文档内容\n"
                "6. 如果无法从文档中确定答案，返回 \"无法确定\""
            )

            user_prompt = (
                f"参考文档：\n{context[:4000]}\n\n"
                f"问题：{question}\n\n"
                f"请给出简短精准的答案（只回答答案本身，不要额外解释）："
            )

            # 检测是否为 DeepSeek 模型
            is_deepseek = "deepseek" in self.base_url.lower() or "deepseek" in self.model.lower()
            
            api_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": 100,  # 限制为 100 tokens，强制简短答案
            }

            # DeepSeek 专用：禁用思考模式
            if is_deepseek:
                api_kwargs["extra_body"] = {"enable_thinking": False}

            response = client.chat.completions.create(**api_kwargs)

            answer = response.choices[0].message.content
            if not answer:
                return ""

            # 去除可能的思考标签
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()

            return answer.strip()

        except Exception as e:
            logger.warning(f"Chat API 生成答案失败: {e}")
            return ""

    # ==========================================
    # 辅助方法
    # ==========================================

    def _collect_supporting_facts(self, reasoning_result: ReasoningResult) -> List[str]:
        """
        收集支撑事实文本

        Args:
            reasoning_result: 推理结果

        Returns:
            支撑事实文本列表
        """
        facts = []
        seen = set()

        # 从推理链的各步骤中收集
        if reasoning_result.reasoning_chain:
            for step in reasoning_result.reasoning_chain.steps:
                if step.result and step.result not in seen:
                    facts.append(f"[Hop {step.hop_id}] {step.result}")
                    seen.add(step.result)

        # 从高分检索结果中收集
        for r in reasoning_result.retrieved_results[:5]:
            if r.text and r.text not in seen:
                text_preview = r.text[:200] + "..." if len(r.text) > 200 else r.text
                facts.append(text_preview)
                seen.add(r.text)

        return facts

    def _build_reasoning_path(self, reasoning_result: ReasoningResult) -> List[str]:
        """
        构建人类可读的推理路径

        Args:
            reasoning_result: 推理结果

        Returns:
            推理路径描述列表
        """
        path = []

        if reasoning_result.reasoning_chain:
            for step in reasoning_result.reasoning_chain.steps:
                path_entry = f"Step {step.hop_id}: {step.query}"
                if step.result:
                    result_preview = (
                        step.result[:80] + "..."
                        if len(step.result) > 80 else step.result
                    )
                    path_entry += f" → {result_preview}"
                if step.expert_id:
                    path_entry += f" (via {step.expert_id})"
                path.append(path_entry)

        return path

    def _generate_explanation(self, question: str, answer: str,
                               reasoning_path: List[str]) -> str:
        """
        生成推理过程的自然语言解释

        Args:
            question: 问题
            answer: 答案
            reasoning_path: 推理路径

        Returns:
            解释文本
        """
        if not reasoning_path:
            return f"Based on retrieved information, the answer to '{question}' is '{answer}'."

        parts = [f"To answer '{question}':"]
        for step_desc in reasoning_path:
            parts.append(f"  - {step_desc}")
        parts.append(f"Therefore, the answer is: {answer}")

        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"AnswerGenerator(model='{self.model}', base_url='{self.base_url}')"
