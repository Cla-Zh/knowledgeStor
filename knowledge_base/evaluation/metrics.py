"""
评估指标 (Evaluation Metrics)

提供多跳问答系统常用的评估指标：
- Exact Match (EM): 精确匹配率
- F1 Score: 基于 token 重叠的 F1 分数
- Answer Accuracy: 答案准确率（归一化后匹配）
- Reasoning Path Accuracy: 推理路径准确率

兼容 HotpotQA / 2WikiMultihopQA / MuSiQue 等数据集的评估标准。
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Optional


# ==============================================================================
# 文本归一化
# ==============================================================================


def normalize_answer(text: str) -> str:
    """
    归一化答案文本

    对齐 HotpotQA / SQuAD 官方评估脚本的处理方式：
    1. 转小写
    2. 移除标点符号
    3. 移除冠词 (a, an, the)
    4. 合并多余空格

    Args:
        text: 原始答案文本

    Returns:
        归一化后的文本
    """
    if not text:
        return ""

    # 转小写
    text = text.lower()

    # 移除标点
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 移除冠词
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # 合并多余空格
    text = " ".join(text.split())

    return text.strip()


def _get_tokens(text: str) -> List[str]:
    """将归一化后的文本转为 token 列表"""
    return normalize_answer(text).split()


# ==============================================================================
# 核心指标
# ==============================================================================


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    精确匹配 (Exact Match / EM)

    归一化后完全相同则为 1.0，否则为 0.0。

    Args:
        prediction: 预测答案
        ground_truth: 标准答案

    Returns:
        1.0 或 0.0
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    基于 token 重叠的 F1 分数

    计算预测答案与标准答案之间的 token 级 Precision、Recall、F1。

    Args:
        prediction: 预测答案
        ground_truth: 标准答案

    Returns:
        F1 分数 (0.0 ~ 1.0)
    """
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(ground_truth)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def precision_score(prediction: str, ground_truth: str) -> float:
    """Token 级精确率"""
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(ground_truth)

    if not pred_tokens:
        return 1.0 if not gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    return num_same / len(pred_tokens) if pred_tokens else 0.0


def recall_score(prediction: str, ground_truth: str) -> float:
    """Token 级召回率"""
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(ground_truth)

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    return num_same / len(gold_tokens) if gold_tokens else 0.0


def answer_accuracy(prediction: str, ground_truth: str,
                     aliases: Optional[List[str]] = None) -> float:
    """
    答案准确率

    如果预测与标准答案（或其任一别名）精确匹配，则为 1.0。

    Args:
        prediction: 预测答案
        ground_truth: 标准答案
        aliases: 答案别名列表（如数据集中的 answer_aliases）

    Returns:
        1.0 或 0.0
    """
    if exact_match(prediction, ground_truth) == 1.0:
        return 1.0

    if aliases:
        for alias in aliases:
            if exact_match(prediction, alias) == 1.0:
                return 1.0

    # 宽松匹配：预测是否包含在标准答案中，或反之
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(ground_truth)
    if pred_norm and gold_norm:
        if pred_norm in gold_norm or gold_norm in pred_norm:
            return 0.5  # 部分匹配给 0.5 分

    return 0.0


def reasoning_path_accuracy(predicted_titles: List[str],
                             gold_titles: List[str]) -> float:
    """
    推理路径准确率

    比较预测的支撑事实标题与标准答案的支撑事实标题的重叠度。

    Args:
        predicted_titles: 预测使用的文档标题列表
        gold_titles: 标准答案的支撑事实标题列表

    Returns:
        F1 分数 (0.0 ~ 1.0)
    """
    if not predicted_titles and not gold_titles:
        return 1.0
    if not predicted_titles or not gold_titles:
        return 0.0

    pred_set = set(t.lower().strip() for t in predicted_titles)
    gold_set = set(t.lower().strip() for t in gold_titles)

    common = pred_set & gold_set
    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


# ==============================================================================
# 批量计算
# ==============================================================================


def batch_exact_match(predictions: List[str],
                       ground_truths: List[str]) -> float:
    """批量计算平均 EM"""
    if not predictions:
        return 0.0
    scores = [
        exact_match(p, g) for p, g in zip(predictions, ground_truths)
    ]
    return sum(scores) / len(scores)


def batch_f1_score(predictions: List[str],
                    ground_truths: List[str]) -> float:
    """批量计算平均 F1"""
    if not predictions:
        return 0.0
    scores = [
        f1_score(p, g) for p, g in zip(predictions, ground_truths)
    ]
    return sum(scores) / len(scores)
