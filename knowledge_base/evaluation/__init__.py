"""
评估模块 (Evaluation)
提供系统评估、指标计算和基线对比功能
"""

from evaluation.metrics import (
    exact_match,
    f1_score,
    normalize_answer,
)
from evaluation.evaluator import Evaluator, EvaluationResults

__all__ = [
    'exact_match',
    'f1_score',
    'normalize_answer',
    'Evaluator',
    'EvaluationResults',
]
