"""
问答流程 (QA Pipeline)

编排完整的问答流程，从用户自然语言问题到最终答案。

流程：
1. 问题理解 (QuestionUnderstanding)
2. 多跳推理 (MultiHopReasoner → ExpertRouter → BaseExpert.hybrid_retrieve)
3. 答案生成 (AnswerGenerator)

支持从已构建的知识库自动加载所有专家。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from experts.base_expert import DomainExpert
from experts.expert_router import ExpertRouter
from pipeline.kb_build_pipeline import KBBuildPipeline
from qa_engine.answer_generator import Answer, AnswerGenerator
from qa_engine.multi_hop_reasoner import MultiHopReasoner, ReasoningResult
from qa_engine.question_understanding import QuestionAnalysis, QuestionUnderstanding
from utils.config import Config

logger = logging.getLogger(__name__)


class QAPipeline:
    """
    问答完整流程

    一站式问答：用户输入自然语言问题，系统返回答案 + 推理路径 + 置信度。

    Usage:
        >>> qa = QAPipeline(kb_path="data/knowledge_bases")
        >>> answer = qa.answer("Who is the CEO of the company that made the iPhone?")
        >>> print(answer.text)
        >>> print(answer.confidence)
        >>> print(answer.explanation)
    """

    def __init__(
        self,
        kb_path: Optional[str] = None,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        初始化问答流程

        Args:
            kb_path: 知识库根目录（包含各专家的子目录）
            config_path: 配置文件路径
            config: 配置对象
        """
        # 加载配置
        if config is not None:
            self._config = config
        elif config_path:
            self._config = Config.load(config_path)
        else:
            self._config = Config.load()

        self.kb_path = kb_path or self._config.get(
            "storage.kb_base_path", "./data/knowledge_bases"
        )

        # 加载专家
        self.experts: List[DomainExpert] = []
        self.expert_router: Optional[ExpertRouter] = None

        if os.path.exists(self.kb_path):
            self._load_experts()
        else:
            logger.warning(f"知识库目录不存在: {self.kb_path}，请先构建知识库")

        # 初始化各组件
        self.question_understanding = QuestionUnderstanding(config=self._config)
        self.reasoner: Optional[MultiHopReasoner] = None
        if self.expert_router:
            self.reasoner = MultiHopReasoner(
                expert_router=self.expert_router,
                config=self._config,
            )
        self.answer_generator = AnswerGenerator(config=self._config)

        logger.info(
            f"QAPipeline 初始化完成: {len(self.experts)} 个专家, "
            f"kb_path={self.kb_path}"
        )

    # ==========================================
    # 专家加载
    # ==========================================

    def _load_experts(self):
        """从知识库目录加载所有专家"""
        builder = KBBuildPipeline(config=self._config)
        self.experts = builder.load_experts(self.kb_path)

        if self.experts:
            self.expert_router = ExpertRouter(
                self.experts,
                routing_model=self._config.get(
                    "system.expert_identification_method", "similarity"
                ),
                config=self._config,
            )
            logger.info(f"已加载 {len(self.experts)} 个专家")
        else:
            logger.warning("未加载到任何专家")

    # ==========================================
    # 主接口
    # ==========================================

    def answer(self, question: str) -> Answer:
        """
        完整问答流程

        Args:
            question: 用户自然语言问题

        Returns:
            Answer 对象（含答案文本、置信度、支撑事实、推理路径）
        """
        start_time = time.time()
        question = question.strip()

        if not question:
            return Answer(text="", confidence=0.0, explanation="Empty question.")

        if not self.experts:
            return Answer(
                text="",
                confidence=0.0,
                explanation="No experts loaded. Please build the knowledge base first.",
            )

        logger.info(f"问答开始: '{question[:60]}...'")

        # Step 1: 问题理解
        analysis = self.question_understanding.analyze(question)
        logger.info(
            f"  分析: type={analysis.question_type}, "
            f"hops={analysis.required_hops}, "
            f"strategy={analysis.reasoning_strategy}, "
            f"entities={analysis.key_entities}"
        )

        # Step 2: 推理
        if self.reasoner is None:
            return Answer(
                text="",
                confidence=0.0,
                explanation="Reasoner not initialized.",
            )

        reasoning_result = self.reasoner.reason(analysis)

        # Step 3: 生成答案
        answer = self.answer_generator.generate(analysis, reasoning_result)

        # 补充元数据
        elapsed = time.time() - start_time
        answer.metadata["time_elapsed"] = elapsed
        answer.metadata["question_type"] = analysis.question_type
        answer.metadata["reasoning_strategy"] = analysis.reasoning_strategy
        answer.metadata["total_hops"] = reasoning_result.total_hops

        logger.info(
            f"  答案: '{answer.text[:60]}...', "
            f"conf={answer.confidence:.2f}, "
            f"source={answer.source}, time={elapsed:.2f}s"
        )

        return answer

    # ==========================================
    # 批量问答
    # ==========================================

    def answer_batch(self, questions: List[str],
                      show_progress: bool = True) -> List[Answer]:
        """
        批量问答

        Args:
            questions: 问题列表
            show_progress: 是否显示进度

        Returns:
            答案列表
        """
        answers = []
        total = len(questions)

        for i, question in enumerate(questions):
            try:
                answer = self.answer(question)
            except Exception as e:
                logger.warning(f"问题 {i+1} 失败: {e}")
                answer = Answer(text="", confidence=0.0)

            answers.append(answer)

            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"批量问答进度: {i+1}/{total}")

        return answers

    # ==========================================
    # 交互式问答
    # ==========================================

    def interactive(self):
        """
        交互式问答循环

        在终端中进行问答交互，输入 'quit' 或 'exit' 退出。
        """
        print("\n" + "=" * 50)
        print("NSEQA 交互式问答系统")
        print(f"已加载 {len(self.experts)} 个专家")
        print("输入问题开始问答，输入 'quit' 退出")
        print("=" * 50 + "\n")

        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if question.lower() in ("quit", "exit", "q"):
                print("再见！")
                break

            if not question:
                continue

            answer = self.answer(question)

            print(f"\nAnswer: {answer.text}")
            print(f"Confidence: {answer.confidence:.2f}")
            print(f"Source: {answer.source}")

            if answer.reasoning_path:
                print("Reasoning Path:")
                for step in answer.reasoning_path:
                    print(f"  {step}")

            if answer.explanation:
                print(f"Explanation: {answer.explanation}")

            print()

    # ==========================================
    # 工具方法
    # ==========================================

    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            "num_experts": len(self.experts),
            "kb_path": self.kb_path,
            "experts": [],
        }
        for expert in self.experts:
            stats["experts"].append(expert.get_statistics())
        return stats

    def __repr__(self) -> str:
        return (
            f"QAPipeline(experts={len(self.experts)}, "
            f"kb_path='{self.kb_path}')"
        )
