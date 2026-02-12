"""
神经-符号转换层 (Neuro-Symbolic Bridge)
负责自然语言与符号表示之间的转换：语义解析、查询生成、结果解释
"""

from neuro_symbolic.semantic_parser import SemanticParser, StructuredQuery, HopQuery

__all__ = [
    'SemanticParser',
    'StructuredQuery',
    'HopQuery',
]
