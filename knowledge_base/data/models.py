"""
数据模型定义 (Data Models)

定义系统中所有核心数据结构，包括：
- Entity: 实体（人物、地点、组织等）
- Relation: 关系（实体间的关联）
- Fact: 事实（支撑证据）
- Question: 问题（包含问题文本、类型、答案等）
- QuestionType: 问题类型枚举
- ReasoningChain: 推理链（多跳推理路径）
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ==============================================================================
# 枚举类型
# ==============================================================================


class QuestionType(Enum):
    """问题类型枚举"""

    SINGLE_HOP = "single_hop"           # 单跳问题
    MULTI_HOP = "multi_hop"             # 多跳问题
    COMPARISON = "comparison"           # 比较类问题
    BRIDGE = "bridge"                   # 桥接类问题（中间实体连接）
    COUNTING = "counting"              # 计数类问题
    YES_NO = "yes_no"                  # 是非题
    UNKNOWN = "unknown"                # 未知类型


class EntityType(Enum):
    """实体类型枚举"""

    PERSON = "person"                  # 人物
    LOCATION = "location"              # 地点
    ORGANIZATION = "organization"      # 组织
    EVENT = "event"                    # 事件
    DATE = "date"                      # 日期
    NUMBER = "number"                  # 数值
    WORK = "work"                      # 作品（电影、书籍等）
    OTHER = "other"                    # 其他


# ==============================================================================
# 核心数据模型
# ==============================================================================


@dataclass
class Entity:
    """
    实体模型

    表示知识图谱中的一个节点，如人物、地点、组织等。

    Attributes:
        id: 实体唯一标识符
        name: 实体名称
        type: 实体类型（person, location, organization 等）
        attributes: 实体属性字典（如出生日期、所在地等附加信息）
        aliases: 实体别名列表（用于匹配不同的名称表述）
        source: 实体来源（标明数据来源，便于溯源）
    """

    id: str = ""
    name: str = ""
    type: str = EntityType.OTHER.value
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    source: str = ""

    def __post_init__(self):
        """如果没有提供 id，则自动生成"""
        if not self.id:
            self.id = f"ent_{uuid.uuid4().hex[:12]}"

    def matches(self, name: str) -> bool:
        """
        判断给定名称是否匹配该实体

        Args:
            name: 待匹配的名称

        Returns:
            是否匹配（名称或别名）
        """
        name_lower = name.lower().strip()
        if self.name.lower().strip() == name_lower:
            return True
        return any(alias.lower().strip() == name_lower for alias in self.aliases)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "attributes": self.attributes,
            "aliases": self.aliases,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Entity:
        """从字典反序列化"""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            type=data.get("type", EntityType.OTHER.value),
            attributes=data.get("attributes", {}),
            aliases=data.get("aliases", []),
            source=data.get("source", ""),
        )

    def __repr__(self) -> str:
        return f"Entity(id='{self.id}', name='{self.name}', type='{self.type}')"


@dataclass
class Relation:
    """
    关系模型

    表示两个实体之间的关系（知识图谱中的边）。

    Attributes:
        subject: 主语实体（名称或 ID）
        predicate: 关系谓词（如 "CEO_of", "born_in" 等）
        object: 宾语实体（名称或 ID）
        confidence: 关系置信度（0.0 ~ 1.0）
        source: 关系来源
        attributes: 附加属性（如时间、条件等）
    """

    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 1.0
    source: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_triple(self) -> tuple:
        """返回 (主语, 谓词, 宾语) 三元组"""
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Relation:
        """从字典反序列化"""
        return cls(
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            object=data.get("object", ""),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
            attributes=data.get("attributes", {}),
        )

    def __repr__(self) -> str:
        return (
            f"Relation('{self.subject}' --[{self.predicate}]--> '{self.object}', "
            f"conf={self.confidence:.2f})"
        )


@dataclass
class Fact:
    """
    事实模型

    表示一条支撑事实，包含文本和提取的结构化信息。
    用于问答系统中的证据追溯和推理路径构建。

    Attributes:
        text: 事实的原始文本
        title: 事实来源的文档标题
        entities: 从该事实中提取的实体名称列表
        relations: 从该事实中提取的关系列表
        sentence_idx: 在文档中的句子索引（用于定位）
    """

    text: str = ""
    title: str = ""
    entities: List[str] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    sentence_idx: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "text": self.text,
            "title": self.title,
            "entities": self.entities,
            "relations": [r.to_dict() for r in self.relations],
            "sentence_idx": self.sentence_idx,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Fact:
        """从字典反序列化"""
        return cls(
            text=data.get("text", ""),
            title=data.get("title", ""),
            entities=data.get("entities", []),
            relations=[
                Relation.from_dict(r) for r in data.get("relations", [])
            ],
            sentence_idx=data.get("sentence_idx", -1),
        )

    def __repr__(self) -> str:
        text_preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"Fact(title='{self.title}', text='{text_preview}')"


@dataclass
class Question:
    """
    问题模型

    表示一个完整的问答对，包含问题文本、答案、支撑事实和推理信息。
    兼容 2WikiMultihopQA 和 MuSiQue 两种数据集格式。

    Attributes:
        id: 问题唯一标识符
        text: 问题文本
        question_type: 问题类型（单跳、多跳、比较等）
        answer: 标准答案
        supporting_facts: 支撑事实列表
        reasoning_hops: 推理所需的跳数
        context: 上下文文档列表 [(title, sentences), ...]
        metadata: 附加元数据（如数据集名称、难度等级等）
    """

    id: str = ""
    text: str = ""
    question_type: str = QuestionType.UNKNOWN.value
    answer: str = ""
    supporting_facts: List[Fact] = field(default_factory=list)
    reasoning_hops: int = 1
    context: List[tuple] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """如果没有提供 id，则自动生成"""
        if not self.id:
            self.id = f"q_{uuid.uuid4().hex[:12]}"

    @property
    def is_multi_hop(self) -> bool:
        """判断是否为多跳问题"""
        return self.reasoning_hops > 1

    @property
    def supporting_titles(self) -> List[str]:
        """获取所有支撑事实的文档标题"""
        return list(set(f.title for f in self.supporting_facts if f.title))

    def get_context_by_title(self, title: str) -> Optional[List[str]]:
        """
        根据标题获取对应的上下文句子

        Args:
            title: 文档标题

        Returns:
            句子列表，如果未找到返回 None
        """
        for ctx_title, sentences in self.context:
            if ctx_title == title:
                return sentences
        return None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "text": self.text,
            "question_type": self.question_type,
            "answer": self.answer,
            "supporting_facts": [f.to_dict() for f in self.supporting_facts],
            "reasoning_hops": self.reasoning_hops,
            "context": self.context,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Question:
        """从字典反序列化"""
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            question_type=data.get("question_type", QuestionType.UNKNOWN.value),
            answer=data.get("answer", ""),
            supporting_facts=[
                Fact.from_dict(f) for f in data.get("supporting_facts", [])
            ],
            reasoning_hops=data.get("reasoning_hops", 1),
            context=data.get("context", []),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"Question(id='{self.id}', type='{self.question_type}', "
            f"hops={self.reasoning_hops}, text='{text_preview}')"
        )


# ==============================================================================
# 推理链模型
# ==============================================================================


@dataclass
class ReasoningStep:
    """
    推理步骤

    表示推理链中的单个步骤。

    Attributes:
        hop_id: 跳数编号（从 1 开始）
        query: 该步骤的查询内容
        result: 该步骤的查询结果
        expert_id: 处理该步骤的专家 ID
        supporting_fact: 支撑该步骤的事实
        confidence: 该步骤结果的置信度
    """

    hop_id: int = 0
    query: str = ""
    result: str = ""
    expert_id: str = ""
    supporting_fact: Optional[Fact] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "hop_id": self.hop_id,
            "query": self.query,
            "result": self.result,
            "expert_id": self.expert_id,
            "supporting_fact": self.supporting_fact.to_dict() if self.supporting_fact else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningStep:
        """从字典反序列化"""
        sf_data = data.get("supporting_fact")
        return cls(
            hop_id=data.get("hop_id", 0),
            query=data.get("query", ""),
            result=data.get("result", ""),
            expert_id=data.get("expert_id", ""),
            supporting_fact=Fact.from_dict(sf_data) if sf_data else None,
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class ReasoningChain:
    """
    推理链模型

    表示完整的推理过程，包含多个推理步骤。

    Attributes:
        question_id: 对应的问题 ID
        steps: 推理步骤列表
        final_answer: 最终答案
        total_hops: 总跳数
        is_complete: 推理是否完整完成
    """

    question_id: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    total_hops: int = 0
    is_complete: bool = False

    @property
    def reasoning_path(self) -> List[str]:
        """获取推理路径的文本描述"""
        path = []
        for step in self.steps:
            path.append(
                f"Hop {step.hop_id}: {step.query} -> {step.result}"
            )
        return path

    def add_step(self, step: ReasoningStep):
        """添加一个推理步骤"""
        self.steps.append(step)
        self.total_hops = len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "question_id": self.question_id,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "total_hops": self.total_hops,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningChain:
        """从字典反序列化"""
        return cls(
            question_id=data.get("question_id", ""),
            steps=[
                ReasoningStep.from_dict(s) for s in data.get("steps", [])
            ],
            final_answer=data.get("final_answer", ""),
            total_hops=data.get("total_hops", 0),
            is_complete=data.get("is_complete", False),
        )

    def __repr__(self) -> str:
        return (
            f"ReasoningChain(question_id='{self.question_id}', "
            f"hops={self.total_hops}, complete={self.is_complete})"
        )


# ==============================================================================
# 文档模型（用于向量检索）
# ==============================================================================


@dataclass
class Document:
    """
    文档模型

    表示知识库中的一个文档片段，用于向量检索。

    Attributes:
        id: 文档唯一标识符
        text: 文档文本内容
        title: 文档标题
        entities: 文档中包含的实体列表
        metadata: 附加元数据（如来源专家 ID 等）
    """

    id: str = ""
    text: str = ""
    title: str = ""
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """如果没有提供 id，则自动生成"""
        if not self.id:
            self.id = f"doc_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "text": self.text,
            "title": self.title,
            "entities": self.entities,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Document:
        """从字典反序列化"""
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            title=data.get("title", ""),
            entities=data.get("entities", []),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Document(id='{self.id}', title='{self.title}', text='{text_preview}')"
