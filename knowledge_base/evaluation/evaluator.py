"""
评估器 (Evaluator)

对问答系统进行全面评估，支持：
- 在测试数据集上批量评估
- 按问题类型和跳数分类统计
- 与大模型基线结果对比
- 生成详细评估报告
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data.loader import DatasetLoader
from data.models import Question, QuestionType
from evaluation.metrics import (
    answer_accuracy,
    batch_exact_match,
    batch_f1_score,
    exact_match,
    f1_score,
    reasoning_path_accuracy,
)
from utils.config import Config

logger = logging.getLogger(__name__)


# ==============================================================================
# 评估结果模型
# ==============================================================================


@dataclass
class EvaluationResults:
    """
    评估结果

    Attributes:
        overall: 总体指标
        by_type: 按问题类型分类的指标
        by_hops: 按推理跳数分类的指标
        details: 每个问题的详细结果
        metadata: 评估元数据（数据集信息、耗时等）
    """

    overall: Dict[str, float] = field(default_factory=dict)
    by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_hops: Dict[int, Dict[str, float]] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "by_type": self.by_type,
            "by_hops": {str(k): v for k, v in self.by_hops.items()},
            "metadata": self.metadata,
            "num_details": len(self.details),
        }

    def save(self, path: str):
        """保存评估结果到 JSON"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = self.to_dict()
        data["details"] = self.details
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"评估结果已保存到: {path}")

    def summary(self) -> str:
        """生成文本摘要"""
        lines = ["=" * 50, "评估结果摘要", "=" * 50]

        lines.append(f"\n总体指标 ({self.metadata.get('total_questions', '?')} 个问题):")
        for metric, value in self.overall.items():
            lines.append(f"  {metric}: {value:.4f}")

        if self.by_type:
            lines.append("\n按问题类型:")
            for q_type, metrics in self.by_type.items():
                count = metrics.get("count", 0)
                em = metrics.get("exact_match", 0)
                f1 = metrics.get("f1_score", 0)
                lines.append(f"  {q_type} (n={count}): EM={em:.4f}, F1={f1:.4f}")

        if self.by_hops:
            lines.append("\n按推理跳数:")
            for hops, metrics in sorted(self.by_hops.items()):
                count = metrics.get("count", 0)
                em = metrics.get("exact_match", 0)
                f1 = metrics.get("f1_score", 0)
                lines.append(f"  {hops}-hop (n={count}): EM={em:.4f}, F1={f1:.4f}")

        if "total_time" in self.metadata:
            lines.append(f"\n总耗时: {self.metadata['total_time']:.1f}s")
        if "avg_time_per_question" in self.metadata:
            lines.append(f"平均每题: {self.metadata['avg_time_per_question']:.2f}s")

        lines.append("=" * 50)
        return "\n".join(lines)

    def __repr__(self) -> str:
        em = self.overall.get("exact_match", 0)
        f1 = self.overall.get("f1_score", 0)
        n = self.metadata.get("total_questions", 0)
        return f"EvaluationResults(n={n}, EM={em:.4f}, F1={f1:.4f})"


# ==============================================================================
# 评估器
# ==============================================================================


class Evaluator:
    """
    系统评估器

    在测试数据集上运行问答系统，计算各项评估指标。

    Usage:
        >>> from pipeline.qa_pipeline import QAPipeline
        >>> qa = QAPipeline(kb_path="data/knowledge_bases")
        >>> evaluator = Evaluator(qa)
        >>> results = evaluator.evaluate("data/raw/2wikimultihopqa/test.json")
        >>> print(results.summary())
    """

    def __init__(
        self,
        qa_pipeline: Any = None,
        config: Optional[Config] = None,
    ):
        """
        初始化评估器

        Args:
            qa_pipeline: QAPipeline 实例（带 .answer(question) 方法）
            config: 配置对象
        """
        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self._config = config
        self.qa_pipeline = qa_pipeline
        self._loader = DatasetLoader()

        logger.info("Evaluator 初始化完成")

    def set_pipeline(self, qa_pipeline: Any):
        """设置问答流程"""
        self.qa_pipeline = qa_pipeline

    # ==========================================
    # 主评估接口
    # ==========================================

    def evaluate(
        self,
        test_data: str | List[Question],
        dataset_type: Optional[str] = None,
        max_samples: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> EvaluationResults:
        """
        在测试数据集上评估问答系统

        Args:
            test_data: 测试数据文件路径，或 Question 列表
            dataset_type: 数据集类型 ('2wiki', 'musique')
            max_samples: 最大评估样本数
            save_path: 结果保存路径（可选）

        Returns:
            EvaluationResults 评估结果
        """
        if self.qa_pipeline is None:
            raise RuntimeError("未设置 QAPipeline，请先调用 set_pipeline()")

        # 加载测试数据
        if isinstance(test_data, str):
            questions = self._loader.load(test_data, dataset_type, max_samples)
        else:
            questions = test_data
            if max_samples:
                questions = questions[:max_samples]

        logger.info(f"开始评估: {len(questions)} 个问题")
        total_start = time.time()

        # 逐题评估
        all_predictions = []
        all_ground_truths = []
        details = []
        question_times = []

        # 按类型和跳数分组
        type_groups: Dict[str, List[Dict]] = defaultdict(list)
        hop_groups: Dict[int, List[Dict]] = defaultdict(list)

        for i, question in enumerate(questions):
            q_start = time.time()

            try:
                # 调用问答系统
                answer = self.qa_pipeline.answer(question.text)
                prediction = answer.text if hasattr(answer, 'text') else str(answer)
                confidence = answer.confidence if hasattr(answer, 'confidence') else 0.0
            except Exception as e:
                logger.warning(f"问题 {i+1} 回答失败: {e}")
                prediction = ""
                confidence = 0.0

            q_time = time.time() - q_start
            question_times.append(q_time)

            ground_truth = question.answer
            all_predictions.append(prediction)
            all_ground_truths.append(ground_truth)

            # 计算单题指标
            em = exact_match(prediction, ground_truth)
            f1 = f1_score(prediction, ground_truth)
            aliases = question.metadata.get("answer_aliases", [])
            acc = answer_accuracy(prediction, ground_truth, aliases)

            detail = {
                "question_id": question.id,
                "question": question.text,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidence": confidence,
                "exact_match": em,
                "f1_score": f1,
                "accuracy": acc,
                "question_type": question.question_type,
                "reasoning_hops": question.reasoning_hops,
                "time": q_time,
            }
            details.append(detail)

            # 分组
            type_groups[question.question_type].append(detail)
            hop_groups[question.reasoning_hops].append(detail)

            # 进度日志
            if (i + 1) % 50 == 0 or i == len(questions) - 1:
                running_em = batch_exact_match(
                    all_predictions, all_ground_truths
                )
                running_f1 = batch_f1_score(
                    all_predictions, all_ground_truths
                )
                logger.info(
                    f"  [{i+1}/{len(questions)}] "
                    f"EM={running_em:.4f}, F1={running_f1:.4f}, "
                    f"avg_time={sum(question_times)/len(question_times):.2f}s"
                )

        total_time = time.time() - total_start

        # 计算总体指标
        overall = self._compute_metrics(details)
        overall["avg_confidence"] = (
            sum(d["confidence"] for d in details) / len(details) if details else 0.0
        )

        # 按类型计算
        by_type = {}
        for q_type, group in type_groups.items():
            by_type[q_type] = self._compute_metrics(group)

        # 按跳数计算
        by_hops = {}
        for hops, group in hop_groups.items():
            by_hops[hops] = self._compute_metrics(group)

        results = EvaluationResults(
            overall=overall,
            by_type=by_type,
            by_hops=by_hops,
            details=details,
            metadata={
                "total_questions": len(questions),
                "total_time": total_time,
                "avg_time_per_question": total_time / max(len(questions), 1),
            },
        )

        logger.info(f"\n{results.summary()}")

        if save_path:
            results.save(save_path)

        return results

    # ==========================================
    # 基线对比
    # ==========================================

    def compare_with_baseline(
        self,
        our_results: EvaluationResults,
        baseline_results: Dict[str, float],
        baseline_name: str = "Baseline",
    ) -> Dict[str, Any]:
        """
        与基线结果对比

        Args:
            our_results: 我们系统的评估结果
            baseline_results: 基线指标字典 {"exact_match": 0.xx, "f1_score": 0.xx}
            baseline_name: 基线名称

        Returns:
            对比报告字典
        """
        report = {
            "baseline_name": baseline_name,
            "baseline": baseline_results,
            "ours": our_results.overall,
            "comparison": {},
        }

        for metric in ["exact_match", "f1_score"]:
            ours_val = our_results.overall.get(metric, 0.0)
            base_val = baseline_results.get(metric, 0.0)
            diff = ours_val - base_val
            report["comparison"][metric] = {
                "ours": ours_val,
                "baseline": base_val,
                "diff": diff,
                "relative_change": diff / base_val * 100 if base_val > 0 else 0.0,
            }

        # 打印对比
        logger.info(f"\n与 {baseline_name} 对比:")
        for metric, info in report["comparison"].items():
            sign = "+" if info["diff"] >= 0 else ""
            logger.info(
                f"  {metric}: {info['ours']:.4f} vs {info['baseline']:.4f} "
                f"({sign}{info['diff']:.4f}, {sign}{info['relative_change']:.1f}%)"
            )

        return report

    # ==========================================
    # 辅助方法
    # ==========================================

    @staticmethod
    def _compute_metrics(details: List[Dict]) -> Dict[str, float]:
        """从详情列表计算聚合指标"""
        if not details:
            return {"count": 0, "exact_match": 0.0, "f1_score": 0.0, "accuracy": 0.0}

        n = len(details)
        return {
            "count": n,
            "exact_match": sum(d["exact_match"] for d in details) / n,
            "f1_score": sum(d["f1_score"] for d in details) / n,
            "accuracy": sum(d["accuracy"] for d in details) / n,
            "avg_time": sum(d["time"] for d in details) / n,
        }

    def __repr__(self) -> str:
        has_pipeline = self.qa_pipeline is not None
        return f"Evaluator(pipeline={'set' if has_pipeline else 'unset'})"
