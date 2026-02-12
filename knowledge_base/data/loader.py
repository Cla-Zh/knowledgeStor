"""
数据加载器 (Dataset Loader)

负责加载和解析不同格式的多跳问答数据集：
- 2WikiMultihopQA: 基于维基百科的多跳问答数据集
- MuSiQue: 多步推理问答数据集

将原始数据转换为统一的内部数据模型（Question, Entity, Relation, Fact）。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.models import (
    Document,
    Entity,
    EntityType,
    Fact,
    Question,
    QuestionType,
    Relation,
    ReasoningChain,
    ReasoningStep,
)

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    统一数据集加载器

    支持加载 2WikiMultihopQA 和 MuSiQue 数据集，
    将其转换为统一的内部数据结构。

    Usage:
        >>> loader = DatasetLoader()
        >>> questions = loader.load_2wiki("data/raw/2wikimultihopqa/train.json")
        >>> entities = loader.extract_entities_from_questions(questions)
        >>> relations = loader.extract_relations_from_questions(questions)
    """

    # 2WikiMultihopQA 问题类型映射
    _2WIKI_TYPE_MAP = {
        "comparison": QuestionType.COMPARISON.value,
        "bridge_comparison": QuestionType.COMPARISON.value,
        "compositional": QuestionType.MULTI_HOP.value,
        "inference": QuestionType.MULTI_HOP.value,
        "bridge": QuestionType.BRIDGE.value,
    }

    # MuSiQue 问题类型映射（基于跳数）
    _MUSIQUE_HOP_TYPE = {
        1: QuestionType.SINGLE_HOP.value,
        2: QuestionType.MULTI_HOP.value,
        3: QuestionType.MULTI_HOP.value,
        4: QuestionType.MULTI_HOP.value,
    }

    # 常见实体类型关键词（用于简单启发式实体类型推断）
    _ENTITY_TYPE_HINTS = {
        EntityType.PERSON.value: [
            "born", "died", "actor", "actress", "director", "politician",
            "president", "singer", "author", "writer", "artist", "athlete",
            "player", "coach", "scientist", "king", "queen", "prince",
        ],
        EntityType.LOCATION.value: [
            "city", "country", "state", "capital", "located", "continent",
            "region", "province", "district", "island", "mountain", "river",
            "lake", "ocean", "sea",
        ],
        EntityType.ORGANIZATION.value: [
            "company", "university", "school", "college", "founded",
            "headquarters", "corporation", "organization", "institution",
            "agency", "department", "team", "club", "band", "party",
        ],
        EntityType.WORK.value: [
            "film", "movie", "book", "novel", "album", "song", "show",
            "series", "game", "painting", "play", "directed", "starring",
        ],
        EntityType.EVENT.value: [
            "war", "battle", "election", "tournament", "championship",
            "festival", "ceremony", "revolution", "disaster",
        ],
    }

    def __init__(self):
        """初始化数据加载器"""
        self._loaded_questions: List[Question] = []
        self._loaded_entities: List[Entity] = []
        self._loaded_relations: List[Relation] = []

    # ==========================================
    # 2WikiMultihopQA 数据集加载
    # ==========================================

    def load_2wiki(self, path: str, max_samples: Optional[int] = None) -> List[Question]:
        """
        加载 2WikiMultihopQA 数据集

        数据集格式（每条记录）：
        {
            "_id": "...",
            "question": "...",
            "answer": "...",
            "type": "comparison|bridge|compositional|...",
            "supporting_facts": [["title", sent_idx], ...],
            "context": [["title", ["sent1", "sent2", ...]], ...],
            "evidences": [{"sub": ..., "rel": ..., "obj": ...}, ...]
        }

        Args:
            path: 数据集 JSON 文件路径
            max_samples: 最大加载样本数（None 表示全部加载）

        Returns:
            Question 对象列表
        """
        logger.info(f"开始加载 2WikiMultihopQA 数据集: {path}")
        raw_data = self._load_json(path)

        if max_samples is not None:
            raw_data = raw_data[:max_samples]

        questions = []
        for item in raw_data:
            try:
                question = self._parse_2wiki_item(item)
                questions.append(question)
            except Exception as e:
                logger.warning(f"解析 2Wiki 数据项失败 (id={item.get('_id', '?')}): {e}")

        self._loaded_questions.extend(questions)
        logger.info(f"2WikiMultihopQA 加载完成: {len(questions)} 条问题")
        return questions

    def _parse_2wiki_item(self, item: Dict[str, Any]) -> Question:
        """
        解析 2WikiMultihopQA 的单条数据

        Args:
            item: 原始 JSON 数据项

        Returns:
            Question 对象
        """
        # 基本信息
        question_id = item.get("_id", "")
        question_text = item.get("question", "")
        answer = item.get("answer", "")
        q_type = item.get("type", "unknown")

        # 解析问题类型
        question_type = self._2WIKI_TYPE_MAP.get(q_type, QuestionType.UNKNOWN.value)

        # 解析上下文
        context = []
        context_raw = item.get("context", [])
        for ctx_item in context_raw:
            if isinstance(ctx_item, list) and len(ctx_item) >= 2:
                title = ctx_item[0]
                sentences = ctx_item[1] if isinstance(ctx_item[1], list) else [ctx_item[1]]
                context.append((title, sentences))

        # 解析支撑事实
        supporting_facts = []
        sf_raw = item.get("supporting_facts", [])
        for sf_item in sf_raw:
            if isinstance(sf_item, list) and len(sf_item) >= 2:
                title = sf_item[0]
                sent_idx = sf_item[1]

                # 从上下文中找到对应的句子
                text = ""
                for ctx_title, sentences in context:
                    if ctx_title == title and 0 <= sent_idx < len(sentences):
                        text = sentences[sent_idx]
                        break

                supporting_facts.append(Fact(
                    text=text,
                    title=title,
                    sentence_idx=sent_idx,
                ))

        # 从 evidences 中提取关系
        evidences = item.get("evidences", [])
        for evidence in evidences:
            if isinstance(evidence, dict):
                for fact in supporting_facts:
                    rel = Relation(
                        subject=evidence.get("sub", ""),
                        predicate=evidence.get("rel", ""),
                        object=evidence.get("obj", ""),
                        source="2wiki_evidence",
                    )
                    if rel.subject and rel.predicate and rel.object:
                        fact.relations.append(rel)
                        # 同时提取实体名称
                        if rel.subject not in fact.entities:
                            fact.entities.append(rel.subject)
                        if rel.object not in fact.entities:
                            fact.entities.append(rel.object)

        # 计算推理跳数
        reasoning_hops = self._estimate_hops_2wiki(item)

        return Question(
            id=question_id,
            text=question_text,
            question_type=question_type,
            answer=answer,
            supporting_facts=supporting_facts,
            reasoning_hops=reasoning_hops,
            context=context,
            metadata={
                "dataset": "2WikiMultihopQA",
                "original_type": q_type,
                "num_evidences": len(evidences),
            },
        )

    def _estimate_hops_2wiki(self, item: Dict[str, Any]) -> int:
        """
        估计 2WikiMultihopQA 问题的推理跳数

        Args:
            item: 原始数据项

        Returns:
            推理跳数
        """
        evidences = item.get("evidences", [])
        if evidences:
            return len(evidences)

        # 根据支撑事实的标题数估计
        sf_raw = item.get("supporting_facts", [])
        unique_titles = set()
        for sf_item in sf_raw:
            if isinstance(sf_item, list) and len(sf_item) >= 1:
                unique_titles.add(sf_item[0])
        return max(len(unique_titles), 1)

    # ==========================================
    # MilitaryData 数据集加载
    # ==========================================

    def load_military(self, path: str, max_samples: Optional[int] = None) -> List[Question]:
        """
        加载 MilitaryData 数据集

        支持两种格式：
        1. 知识库文章（构建用）: [{"id": ..., "text": "...", "metadata": {...}}, ...]
        2. 测试问题（评估用）: [{"question": "...", "answer": "..."/"[...]"}, ...]

        自动根据数据内容判断格式。当 path 为目录时，加载目录下所有 JSON 文件。

        Args:
            path: 数据文件或目录路径
            max_samples: 最大加载样本数

        Returns:
            Question 对象列表
        """
        logger.info(f"开始加载 MilitaryData 数据集: {path}")

        # 如果是目录，加载目录下所有 JSON 文件
        if os.path.isdir(path):
            all_data = []
            for json_file in sorted(Path(path).glob("*.json")):
                all_data.extend(self._load_json(str(json_file)))
        else:
            all_data = self._load_json(path)

        if not all_data:
            logger.warning(f"未加载到任何数据: {path}")
            return []

        # 自动判断数据格式：文章 vs 测试题
        sample = all_data[0]
        if "question" in sample and "answer" in sample:
            questions = self._parse_military_test_data(all_data, max_samples)
        else:
            questions = self._parse_military_articles(all_data, max_samples)

        self._loaded_questions.extend(questions)
        logger.info(f"MilitaryData 加载完成: {len(questions)} 条数据")
        return questions

    def _parse_military_articles(self, articles: List[Dict[str, Any]],
                                  max_samples: Optional[int] = None) -> List[Question]:
        """
        解析 MilitaryData 知识库文章

        将每篇文章转换为 Question 对象，文章文本作为上下文，
        同时将全文写入 text 字段用于后续领域聚类。

        Args:
            articles: 文章数据列表
            max_samples: 最大样本数

        Returns:
            Question 对象列表
        """
        if max_samples is not None:
            articles = articles[:max_samples]

        questions = []
        for i, article in enumerate(articles):
            try:
                text = article.get("text", "")
                if not text.strip():
                    continue

                # 提取标题
                title = self._extract_title_from_text(text)
                if not title:
                    title = f"military_article_{i}"

                # 将文章按段落切分（用于上下文）
                paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

                question = Question(
                    id=f"military_art_{i}",
                    text=text,  # 全文用于 TF-IDF 聚类
                    question_type=QuestionType.UNKNOWN.value,
                    answer="",
                    context=[(title, paragraphs)],
                    metadata={
                        "dataset": "MilitaryData",
                        "data_type": "article",
                        "original_id": article.get("id", ""),
                        "lang": article.get("metadata", {}).get("lang", "zh-CN"),
                    },
                )
                questions.append(question)
            except Exception as e:
                logger.warning(f"解析 MilitaryData 文章 {i} 失败: {e}")

        logger.info(f"MilitaryData 文章加载完成: {len(questions)} 篇文章")
        return questions

    def _parse_military_test_data(self, test_data: List[Dict[str, Any]],
                                   max_samples: Optional[int] = None) -> List[Question]:
        """
        解析 MilitaryData 测试数据

        将测试题转换为 Question 对象，支持 answer 为字符串或列表。

        Args:
            test_data: 测试数据列表
            max_samples: 最大样本数

        Returns:
            Question 对象列表
        """
        if max_samples is not None:
            test_data = test_data[:max_samples]

        questions = []
        for i, item in enumerate(test_data):
            try:
                question_text = item.get("question", "")
                if not question_text.strip():
                    continue

                raw_answer = item.get("answer", "")

                # 处理答案格式：可能是字符串或列表
                if isinstance(raw_answer, list):
                    answer = raw_answer[0] if raw_answer else ""
                    answer_aliases = raw_answer[1:] if len(raw_answer) > 1 else []
                else:
                    answer = str(raw_answer)
                    answer_aliases = []

                # 提取 chunk_titles（用于文档召回评估）
                chunk_titles = item.get("chunk_titles", [])
                cleaned_chunk_titles = []
                for ct in chunk_titles:
                    if ct.startswith("标题: "):
                        cleaned_chunk_titles.append(ct[len("标题: "):])
                    elif ct.startswith("标题:"):
                        cleaned_chunk_titles.append(ct[len("标题:"):])
                    else:
                        cleaned_chunk_titles.append(ct)

                question = Question(
                    id=f"military_test_{i}",
                    text=question_text,
                    question_type=QuestionType.MULTI_HOP.value,
                    answer=answer,
                    reasoning_hops=3,  # 军事领域多跳问题估计
                    metadata={
                        "dataset": "MilitaryData",
                        "data_type": "test",
                        "answer_aliases": answer_aliases,
                        "chunk_titles": cleaned_chunk_titles,
                        "description": item.get("description", ""),
                    },
                )
                questions.append(question)
            except Exception as e:
                logger.warning(f"解析 MilitaryData 测试题 {i} 失败: {e}")

        logger.info(f"MilitaryData 测试集加载完成: {len(questions)} 个问题")
        return questions

    @staticmethod
    def _extract_title_from_text(text: str) -> str:
        """
        从文章文本中提取标题

        支持 "标题: XXX" 和 "标题:XXX" 格式。

        Args:
            text: 文章全文

        Returns:
            提取到的标题，如果未找到返回空字符串
        """
        if not text:
            return ""

        first_line = text.split("\n")[0].strip()

        # 匹配 "标题:" 或 "标题: " 前缀
        if first_line.startswith("标题:"):
            title = first_line[len("标题:"):].strip()
            if title:
                return title

        # 如果第一行较短（可能本身就是标题）
        if 0 < len(first_line) <= 60:
            return first_line

        return ""

    # ==========================================
    # MuSiQue 数据集加载
    # ==========================================

    def load_musique(self, path: str, max_samples: Optional[int] = None) -> List[Question]:
        """
        加载 MuSiQue 数据集

        数据集格式（每条记录）：
        {
            "id": "...",
            "question": "...",
            "answer": "...",
            "answer_aliases": [...],
            "answerable": true/false,
            "paragraphs": [
                {"idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": true/false},
                ...
            ],
            "question_decomposition": [
                {"id": 0, "question": "...", "answer": "...", "paragraph_support_idx": 0},
                ...
            ]
        }

        Args:
            path: 数据集 JSONL 文件路径
            max_samples: 最大加载样本数

        Returns:
            Question 对象列表
        """
        logger.info(f"开始加载 MuSiQue 数据集: {path}")
        raw_data = self._load_jsonl(path)

        if max_samples is not None:
            raw_data = raw_data[:max_samples]

        questions = []
        for item in raw_data:
            try:
                question = self._parse_musique_item(item)
                questions.append(question)
            except Exception as e:
                logger.warning(f"解析 MuSiQue 数据项失败 (id={item.get('id', '?')}): {e}")

        self._loaded_questions.extend(questions)
        logger.info(f"MuSiQue 加载完成: {len(questions)} 条问题")
        return questions

    def _parse_musique_item(self, item: Dict[str, Any]) -> Question:
        """
        解析 MuSiQue 的单条数据

        Args:
            item: 原始 JSON 数据项

        Returns:
            Question 对象
        """
        question_id = str(item.get("id", ""))
        question_text = item.get("question", "")
        answer = item.get("answer", "")

        # 解析段落（上下文）
        context = []
        paragraphs = item.get("paragraphs", [])
        for para in paragraphs:
            title = para.get("title", "")
            text = para.get("paragraph_text", "")
            # 将段落拆分为句子
            sentences = self._split_sentences(text)
            context.append((title, sentences))

        # 解析支撑事实
        supporting_facts = []
        for para in paragraphs:
            if para.get("is_supporting", False):
                title = para.get("title", "")
                text = para.get("paragraph_text", "")
                supporting_facts.append(Fact(
                    text=text,
                    title=title,
                ))

        # 解析问题分解（MuSiQue 特有的结构化推理链）
        decomposition = item.get("question_decomposition", [])
        reasoning_hops = max(len(decomposition), 1)

        # 确定问题类型
        question_type = self._MUSIQUE_HOP_TYPE.get(
            reasoning_hops, QuestionType.MULTI_HOP.value
        )

        return Question(
            id=question_id,
            text=question_text,
            question_type=question_type,
            answer=answer,
            supporting_facts=supporting_facts,
            reasoning_hops=reasoning_hops,
            context=context,
            metadata={
                "dataset": "MuSiQue",
                "answerable": item.get("answerable", True),
                "answer_aliases": item.get("answer_aliases", []),
                "decomposition": decomposition,
                "num_paragraphs": len(paragraphs),
            },
        )

    # ==========================================
    # 通用数据集加载（自动识别格式）
    # ==========================================

    def load(self, path: str, dataset_type: Optional[str] = None,
             max_samples: Optional[int] = None) -> List[Question]:
        """
        通用加载入口，自动识别或手动指定数据集类型

        Args:
            path: 数据集文件或目录路径
            dataset_type: 数据集类型 - '2wiki', 'musique'，None 则自动检测
            max_samples: 最大样本数

        Returns:
            Question 对象列表
        """
        if os.path.isdir(path):
            return self._load_directory(path, dataset_type, max_samples)

        if dataset_type is None:
            dataset_type = self._detect_dataset_type(path)

        if dataset_type == "2wiki":
            return self.load_2wiki(path, max_samples)
        elif dataset_type == "musique":
            return self.load_musique(path, max_samples)
        elif dataset_type == "military":
            return self.load_military(path, max_samples)
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")

    def _load_directory(self, dir_path: str, dataset_type: Optional[str],
                        max_samples: Optional[int]) -> List[Question]:
        """加载目录下的所有数据文件"""
        all_questions = []
        data_dir = Path(dir_path)

        # 查找 JSON 和 JSONL 文件
        data_files = list(data_dir.glob("*.json")) + list(data_dir.glob("*.jsonl"))

        for file_path in sorted(data_files):
            try:
                questions = self.load(str(file_path), dataset_type, max_samples)
                all_questions.extend(questions)
                if max_samples and len(all_questions) >= max_samples:
                    all_questions = all_questions[:max_samples]
                    break
            except Exception as e:
                logger.warning(f"加载文件 {file_path} 失败: {e}")

        return all_questions

    def _detect_dataset_type(self, path: str) -> str:
        """
        自动检测数据集类型

        通过读取文件中的第一条记录来判断格式。

        Args:
            path: 文件路径

        Returns:
            数据集类型标识符
        """
        # 尝试 JSONL 格式（MuSiQue 常用 JSONL）
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    sample = json.loads(first_line)
                    if "question_decomposition" in sample or "paragraphs" in sample:
                        return "musique"
                    if "supporting_facts" in sample and "context" in sample:
                        return "2wiki"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # 尝试 JSON 数组格式（2Wiki 常用 JSON）
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    if "supporting_facts" in sample:
                        return "2wiki"
                    if "question_decomposition" in sample:
                        return "musique"
                    # MilitaryData 文章格式: {"id": ..., "text": "...", "metadata": {...}}
                    if "text" in sample and "metadata" in sample and "question" not in sample:
                        return "military"
                    # MilitaryData 测试格式: {"question": "...", "answer": "..."}
                    if ("question" in sample and "answer" in sample
                            and "supporting_facts" not in sample
                            and "question_decomposition" not in sample):
                        return "military"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        logger.warning(f"无法自动检测数据集类型，默认使用 2wiki: {path}")
        return "2wiki"

    # ==========================================
    # 实体和关系提取
    # ==========================================

    def extract_entities(self, item: Dict[str, Any]) -> List[Entity]:
        """
        从单条原始数据中提取实体

        Args:
            item: 原始数据项

        Returns:
            实体列表
        """
        entities = []
        seen_names = set()

        # 从 evidences/关系中提取
        evidences = item.get("evidences", [])
        for evidence in evidences:
            if isinstance(evidence, dict):
                for key in ("sub", "obj"):
                    name = evidence.get(key, "")
                    if name and name not in seen_names:
                        entity = Entity(
                            name=name,
                            type=self._infer_entity_type(name, evidence),
                            source="evidence",
                        )
                        entities.append(entity)
                        seen_names.add(name)

        # 从上下文标题中提取
        context = item.get("context", [])
        for ctx_item in context:
            if isinstance(ctx_item, list) and len(ctx_item) >= 1:
                title = ctx_item[0]
                if title and title not in seen_names:
                    entity = Entity(
                        name=title,
                        type=self._infer_entity_type_from_context(
                            title, ctx_item[1] if len(ctx_item) > 1 else []
                        ),
                        source="context_title",
                    )
                    entities.append(entity)
                    seen_names.add(title)

        # 从 paragraphs 中提取（MuSiQue 格式）
        paragraphs = item.get("paragraphs", [])
        for para in paragraphs:
            title = para.get("title", "")
            if title and title not in seen_names:
                entity = Entity(
                    name=title,
                    type=self._infer_entity_type_from_context(
                        title, [para.get("paragraph_text", "")]
                    ),
                    source="paragraph_title",
                )
                entities.append(entity)
                seen_names.add(title)

        return entities

    def extract_relations(self, item: Dict[str, Any]) -> List[Relation]:
        """
        从单条原始数据中提取关系

        Args:
            item: 原始数据项

        Returns:
            关系列表
        """
        relations = []

        # 从 evidences 中直接提取（2Wiki 格式）
        evidences = item.get("evidences", [])
        for evidence in evidences:
            if isinstance(evidence, dict):
                subject = evidence.get("sub", "")
                predicate = evidence.get("rel", "")
                obj = evidence.get("obj", "")
                if subject and predicate and obj:
                    relations.append(Relation(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=1.0,
                        source="evidence",
                    ))

        # 从问题分解中推断关系（MuSiQue 格式）
        decomposition = item.get("question_decomposition", [])
        for step in decomposition:
            if isinstance(step, dict):
                sub_question = step.get("question", "")
                sub_answer = step.get("answer", "")
                if sub_question and sub_answer:
                    # 从子问题中启发式提取关系
                    extracted = self._extract_relation_from_sub_question(
                        sub_question, sub_answer
                    )
                    if extracted:
                        relations.append(extracted)

        return relations

    def extract_entities_from_questions(self, questions: List[Question]) -> List[Entity]:
        """
        从已解析的 Question 列表中提取所有唯一实体

        Args:
            questions: 问题列表

        Returns:
            去重后的实体列表
        """
        entities = []
        seen_names = set()

        for question in questions:
            for fact in question.supporting_facts:
                # 从事实的实体列表中提取
                for ent_name in fact.entities:
                    if ent_name not in seen_names:
                        entities.append(Entity(
                            name=ent_name,
                            source=f"question_{question.id}",
                        ))
                        seen_names.add(ent_name)

                # 从事实的关系中提取
                for rel in fact.relations:
                    for name in (rel.subject, rel.object):
                        if name and name not in seen_names:
                            entities.append(Entity(
                                name=name,
                                source=f"question_{question.id}",
                            ))
                            seen_names.add(name)

            # 从上下文标题中提取
            for title, _ in question.context:
                if title and title not in seen_names:
                    entities.append(Entity(
                        name=title,
                        source=f"context_{question.id}",
                    ))
                    seen_names.add(title)

        logger.info(f"从 {len(questions)} 个问题中提取了 {len(entities)} 个唯一实体")
        return entities

    def extract_relations_from_questions(self, questions: List[Question]) -> List[Relation]:
        """
        从已解析的 Question 列表中提取所有关系

        Args:
            questions: 问题列表

        Returns:
            关系列表
        """
        relations = []

        for question in questions:
            for fact in question.supporting_facts:
                relations.extend(fact.relations)

        logger.info(f"从 {len(questions)} 个问题中提取了 {len(relations)} 条关系")
        return relations

    def build_documents(self, questions: List[Question]) -> List[Document]:
        """
        将问题的上下文转换为文档列表（用于向量索引构建）

        每个上下文段落（title + sentences）转换为一个 Document。

        Args:
            questions: 问题列表

        Returns:
            文档列表
        """
        documents = []
        seen_texts = set()

        for question in questions:
            for title, sentences in question.context:
                # 将句子列表合并为完整文本
                full_text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                # 去重
                text_key = f"{title}::{full_text[:100]}"
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                # 收集该段落涉及的实体
                paragraph_entities = [title]
                for fact in question.supporting_facts:
                    if fact.title == title:
                        paragraph_entities.extend(fact.entities)

                documents.append(Document(
                    text=full_text,
                    title=title,
                    entities=list(set(paragraph_entities)),
                    metadata={
                        "question_id": question.id,
                        "is_supporting": title in question.supporting_titles,
                    },
                ))

        logger.info(f"构建了 {len(documents)} 个文档")
        return documents

    def build_reasoning_chain(self, item: Dict[str, Any]) -> ReasoningChain:
        """
        从原始数据构建推理链

        Args:
            item: 原始数据项

        Returns:
            推理链对象
        """
        chain = ReasoningChain(
            question_id=item.get("_id", item.get("id", "")),
        )

        # 从 evidences 构建（2Wiki 格式）
        evidences = item.get("evidences", [])
        for i, evidence in enumerate(evidences):
            if isinstance(evidence, dict):
                step = ReasoningStep(
                    hop_id=i + 1,
                    query=f"{evidence.get('sub', '')} {evidence.get('rel', '')} ?",
                    result=evidence.get("obj", ""),
                    confidence=1.0,
                )
                chain.add_step(step)

        # 从 question_decomposition 构建（MuSiQue 格式）
        decomposition = item.get("question_decomposition", [])
        for step_data in decomposition:
            if isinstance(step_data, dict):
                step = ReasoningStep(
                    hop_id=step_data.get("id", 0) + 1,
                    query=step_data.get("question", ""),
                    result=step_data.get("answer", ""),
                    confidence=1.0,
                )
                chain.add_step(step)

        chain.final_answer = item.get("answer", "")
        chain.is_complete = len(chain.steps) > 0

        return chain

    # ==========================================
    # 辅助方法
    # ==========================================

    def _infer_entity_type(self, name: str, evidence: Dict) -> str:
        """从 evidence 推断实体类型"""
        relation = evidence.get("rel", "").lower()

        for etype, keywords in self._ENTITY_TYPE_HINTS.items():
            if any(kw in relation for kw in keywords):
                return etype

        return EntityType.OTHER.value

    def _infer_entity_type_from_context(self, title: str,
                                         sentences: List[str]) -> str:
        """从上下文推断实体类型"""
        text = (title + " " + " ".join(sentences)).lower()

        type_scores = {}
        for etype, keywords in self._ENTITY_TYPE_HINTS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                type_scores[etype] = score

        if type_scores:
            return max(type_scores, key=type_scores.get)

        return EntityType.OTHER.value

    def _extract_relation_from_sub_question(self, question: str,
                                             answer: str) -> Optional[Relation]:
        """
        从子问题和答案中启发式提取关系

        简单策略：将问题中的关键短语作为谓词
        """
        # 移除常见疑问词
        q_clean = question.lower().strip("?").strip()
        for prefix in ["what is the", "who is the", "where is the",
                       "when was the", "what was the", "who was the",
                       "in which", "what", "who", "where", "when", "which"]:
            if q_clean.startswith(prefix):
                q_clean = q_clean[len(prefix):].strip()
                break

        # 查找介词 "of" 来分割关系和实体
        if " of " in q_clean:
            parts = q_clean.split(" of ", 1)
            predicate = parts[0].strip().replace(" ", "_")
            subject = parts[1].strip()
            if predicate and subject and answer:
                return Relation(
                    subject=subject,
                    predicate=predicate,
                    object=answer,
                    confidence=0.8,
                    source="decomposition",
                )

        return None

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        简单的句子切分

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        if not text:
            return []
        # 使用正则按句号、问号、感叹号切分
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _load_json(path: str) -> List[Dict]:
        """加载 JSON 文件"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"JSON 文件格式不正确: {path}")

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        """加载 JSONL（每行一个 JSON）文件"""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONL 第 {line_num} 行解析失败: {e}")
        return data

    def get_statistics(self, questions: List[Question]) -> Dict[str, Any]:
        """
        获取数据集统计信息

        Args:
            questions: 问题列表

        Returns:
            统计信息字典
        """
        type_counts = {}
        hop_counts = {}
        total_facts = 0
        total_context = 0

        for q in questions:
            # 问题类型统计
            type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
            # 跳数统计
            hop_counts[q.reasoning_hops] = hop_counts.get(q.reasoning_hops, 0) + 1
            # 支撑事实统计
            total_facts += len(q.supporting_facts)
            # 上下文统计
            total_context += len(q.context)

        return {
            "total_questions": len(questions),
            "question_types": type_counts,
            "hop_distribution": hop_counts,
            "avg_supporting_facts": total_facts / max(len(questions), 1),
            "avg_context_paragraphs": total_context / max(len(questions), 1),
        }

    def __repr__(self) -> str:
        return (
            f"DatasetLoader(loaded_questions={len(self._loaded_questions)}, "
            f"loaded_entities={len(self._loaded_entities)}, "
            f"loaded_relations={len(self._loaded_relations)})"
        )
