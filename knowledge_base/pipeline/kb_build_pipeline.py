"""
知识库构建流程 (KB Build Pipeline)

编排完整的知识库构建流程，将原始数据集转化为可用的分布式专家知识库。

完整流程：
1. 加载数据集（2WikiMultihopQA / MuSiQue）
2. 提取实体和关系
3. 自动识别专家领域
4. 将数据分配到各专家
5. 为每个专家构建向量索引
6. 为每个专家构建知识图谱
7. 保存所有索引、图谱和元数据

输出结构：
data/knowledge_bases/
├── expert_0/
│   ├── vectors/          # 向量索引
│   ├── graph/            # 知识图谱
│   └── metadata.json     # 专家元数据
├── expert_1/
│   └── ...
└── build_info.json       # 整体构建信息
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.loader import DatasetLoader
from data.models import Document, Entity, Question, Relation
from kb_builder.expert_identifier import ExpertIdentifier
from kb_builder.graph_builder import KnowledgeGraphBuilder
from kb_builder.models import ExpertDomain, KnowledgeGraph, VectorIndex
from kb_builder.triple_extractor import TripleExtractor
from kb_builder.vector_builder import VectorIndexBuilder
from experts.base_expert import DomainExpert
from utils.config import Config

logger = logging.getLogger(__name__)


class KBBuildPipeline:
    """
    知识库构建完整流程

    将原始数据集一站式转化为分布式专家知识库系统。

    Usage:
        >>> pipeline = KBBuildPipeline(config_path="config.yaml")
        >>> experts = pipeline.run("data/raw/2wikimultihopqa/train.json", dataset_type="2wiki")
        >>> # 构建完成后，知识库保存在 data/knowledge_bases/ 下
        >>> # 返回构建好的专家列表，可直接用于问答
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        初始化知识库构建流程

        Args:
            config_path: 配置文件路径
            config: 配置对象（如果同时提供 config_path 和 config，优先使用 config）
        """
        if config is not None:
            self._config = config
        elif config_path:
            self._config = Config.load(config_path)
        else:
            self._config = Config.load()

        # 输出路径
        self.kb_base_path = self._config.get("storage.kb_base_path", "./data/knowledge_bases")

        # 子组件（延迟初始化，确保 config 已就绪）
        self._loader: Optional[DatasetLoader] = None
        self._identifier: Optional[ExpertIdentifier] = None
        self._vector_builder: Optional[VectorIndexBuilder] = None
        self._graph_builder: Optional[KnowledgeGraphBuilder] = None
        self._triple_extractor: Optional[TripleExtractor] = None

        # 构建过程中的中间数据
        self._questions: List[Question] = []
        self._domains: List[ExpertDomain] = []
        self._expert_questions: Dict[str, List[Question]] = {}
        self._experts: List[DomainExpert] = []
        self._build_info: Dict[str, Any] = {}

        logger.info(
            f"KBBuildPipeline 初始化: output={self.kb_base_path}"
        )

    # ==========================================
    # 组件延迟初始化
    # ==========================================

    @property
    def loader(self) -> DatasetLoader:
        if self._loader is None:
            self._loader = DatasetLoader()
        return self._loader

    @property
    def identifier(self) -> ExpertIdentifier:
        if self._identifier is None:
            self._identifier = ExpertIdentifier(config=self._config)
        return self._identifier

    @property
    def vector_builder(self) -> VectorIndexBuilder:
        if self._vector_builder is None:
            self._vector_builder = VectorIndexBuilder(config=self._config)
        return self._vector_builder

    @property
    def graph_builder(self) -> KnowledgeGraphBuilder:
        if self._graph_builder is None:
            self._graph_builder = KnowledgeGraphBuilder(config=self._config)
        return self._graph_builder

    @property
    def triple_extractor(self) -> TripleExtractor:
        if self._triple_extractor is None:
            self._triple_extractor = TripleExtractor(config=self._config)
        return self._triple_extractor

    # ==========================================
    # 主流程
    # ==========================================

    def run(
        self,
        dataset_path: str,
        dataset_type: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> List[DomainExpert]:
        """
        执行完整的知识库构建流程

        Args:
            dataset_path: 数据集文件或目录路径
            dataset_type: 数据集类型 - '2wiki', 'musique'（None 则自动检测）
            max_samples: 最大样本数（用于调试时限制数据量）

        Returns:
            构建好的专家列表

        Raises:
            FileNotFoundError: 数据集路径不存在
            RuntimeError: 构建过程中出现不可恢复的错误
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        total_start = time.time()
        logger.info("=" * 60)
        logger.info("开始知识库构建流程")
        logger.info("=" * 60)

        try:
            # Step 1: 加载数据集
            self._step_load_data(dataset_path, dataset_type, max_samples)

            # Step 2: 识别专家领域
            self._step_identify_domains()

            # Step 3: 分配数据到各专家
            self._step_assign_data()

            # Step 4: 为每个专家构建知识库
            self._step_build_expert_kbs()

            # Step 5: 保存构建信息
            self._step_save_build_info(dataset_path, total_start)

        except Exception as e:
            logger.error(f"知识库构建失败: {e}", exc_info=True)
            raise

        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info(
            f"知识库构建完成! 耗时: {total_time:.1f}s, "
            f"专家数: {len(self._experts)}"
        )
        logger.info(f"输出路径: {self.kb_base_path}")
        logger.info("=" * 60)

        return self._experts

    # ==========================================
    # 各步骤实现
    # ==========================================

    def _step_load_data(self, dataset_path: str,
                         dataset_type: Optional[str],
                         max_samples: Optional[int]):
        """Step 1: 加载数据集"""
        step_start = time.time()
        logger.info("-" * 40)
        logger.info("Step 1/5: 加载数据集")
        logger.info("-" * 40)

        self._questions = self.loader.load(
            dataset_path,
            dataset_type=dataset_type,
            max_samples=max_samples,
        )

        # 统计信息
        stats = self.loader.get_statistics(self._questions)
        logger.info(f"  加载完成: {stats['total_questions']} 个问题")
        logger.info(f"  问题类型: {stats['question_types']}")
        logger.info(f"  跳数分布: {stats['hop_distribution']}")
        logger.info(f"  平均支撑事实数: {stats['avg_supporting_facts']:.2f}")
        logger.info(f"  耗时: {time.time() - step_start:.1f}s")

        self._build_info["dataset_stats"] = stats

    def _step_identify_domains(self):
        """Step 2: 识别专家领域"""
        step_start = time.time()
        logger.info("-" * 40)
        logger.info("Step 2/5: 识别专家领域")
        logger.info("-" * 40)

        self._domains = self.identifier.identify_domains(self._questions)

        logger.info(f"  识别出 {len(self._domains)} 个专家领域:")
        for domain in self._domains:
            logger.info(
                f"    {domain.id}: {domain.name} "
                f"(types={domain.entity_types[:3]}, keywords={domain.keywords[:5]})"
            )
        logger.info(f"  耗时: {time.time() - step_start:.1f}s")

        self._build_info["domains"] = [d.to_dict() for d in self._domains]

    def _step_assign_data(self):
        """Step 3: 分配数据到各专家"""
        step_start = time.time()
        logger.info("-" * 40)
        logger.info("Step 3/5: 分配数据到各专家")
        logger.info("-" * 40)

        self._expert_questions = self.identifier.assign_data_to_experts(
            self._questions, self._domains
        )

        assignment_info = {}
        for expert_id, questions in self._expert_questions.items():
            logger.info(f"  {expert_id}: {len(questions)} 个问题")
            assignment_info[expert_id] = len(questions)
        logger.info(f"  耗时: {time.time() - step_start:.1f}s")

        self._build_info["assignment"] = assignment_info

    def _step_build_expert_kbs(self):
        """Step 4: 为每个专家构建知识库"""
        step_start = time.time()
        logger.info("-" * 40)
        logger.info("Step 4/5: 构建专家知识库")
        logger.info("-" * 40)

        self._experts = []

        for domain in self._domains:
            expert_start = time.time()
            expert_id = domain.id
            questions = self._expert_questions.get(expert_id, [])

            if not questions:
                logger.warning(f"  {expert_id}: 没有分配数据，跳过")
                continue

            logger.info(f"  构建专家 '{expert_id}' ({domain.name})...")

            # 创建专家实例
            expert_path = os.path.join(self.kb_base_path, expert_id)
            expert = DomainExpert(
                expert_id=expert_id,
                domain=domain,
                config=self._config,
            )

            # 4a: 构建文档并创建向量索引
            logger.info(f"    4a. 构建向量索引...")
            vector_index = self._build_vector_index_for_expert(questions)
            expert.set_vector_index(vector_index)

            # 4b: 提取实体关系并构建知识图谱
            logger.info(f"    4b. 构建知识图谱...")
            knowledge_graph = self._build_knowledge_graph_for_expert(questions)
            expert.set_knowledge_graph(knowledge_graph)

            # 4c: 保存专家知识库
            logger.info(f"    4c. 保存知识库到 {expert_path}...")
            expert.save(expert_path)

            self._experts.append(expert)

            expert_time = time.time() - expert_start
            logger.info(
                f"    完成! 向量={vector_index.size}, "
                f"实体={knowledge_graph.num_entities}, "
                f"关系={knowledge_graph.num_relations}, "
                f"耗时={expert_time:.1f}s"
            )

        logger.info(f"  所有专家构建完成, 耗时: {time.time() - step_start:.1f}s")

    def _step_save_build_info(self, dataset_path: str, total_start: float):
        """Step 5: 保存构建信息"""
        logger.info("-" * 40)
        logger.info("Step 5/5: 保存构建信息")
        logger.info("-" * 40)

        os.makedirs(self.kb_base_path, exist_ok=True)

        self._build_info.update({
            "dataset_path": dataset_path,
            "n_experts": len(self._experts),
            "total_time_seconds": time.time() - total_start,
            "experts": [],
        })

        for expert in self._experts:
            self._build_info["experts"].append(expert.get_statistics())

        # 保存到文件
        info_path = os.path.join(self.kb_base_path, "build_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(self._build_info, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"  构建信息已保存到: {info_path}")

    # ==========================================
    # 向量索引构建
    # ==========================================

    def _build_vector_index_for_expert(self,
                                        questions: List[Question]) -> VectorIndex:
        """
        为单个专家构建向量索引

        Args:
            questions: 分配给该专家的问题列表

        Returns:
            VectorIndex 实例
        """
        # 从问题中提取文档
        documents = self.loader.build_documents(questions)

        if not documents:
            logger.warning("没有文档可用于构建向量索引")
            return VectorIndex()

        # 构建索引
        index = self.vector_builder.build_index_from_documents(
            documents, show_progress=True
        )

        return index

    # ==========================================
    # 知识图谱构建
    # ==========================================

    def _build_knowledge_graph_for_expert(self,
                                           questions: List[Question]) -> KnowledgeGraph:
        """
        为单个专家构建知识图谱

        流程：
        1. 先从结构化数据（evidences / supporting_facts）中提取实体和关系
        2. 如果结构化关系为空（如 MilitaryData），使用 LLM 自动从文本中抽取三元组
        3. 合并所有实体和关系，构建图谱

        Args:
            questions: 分配给该专家的问题列表

        Returns:
            KnowledgeGraph 实例
        """
        # Step 1: 尝试从结构化数据中提取实体和关系
        structured_entities = self.loader.extract_entities_from_questions(questions)
        structured_relations = self.loader.extract_relations_from_questions(questions)
        logger.info(
            f"    结构化数据提取: {len(structured_entities)} 个实体, "
            f"{len(structured_relations)} 条关系"
        )

        # Step 2: 决定实体/关系来源
        if structured_relations:
            # 有结构化关系（2Wiki/MuSiQue），直接使用
            entities = structured_entities
            relations = structured_relations
        elif self.triple_extractor.enabled:
            # 无结构化关系（MilitaryData 等纯文本），完全由 LLM 抽取
            # 不使用标题实体，因为文章标题是整句话，不是有效的图谱节点
            logger.info(
                f"    结构化关系为空，丢弃标题实体，"
                f"启用 LLM 三元组抽取 (model={self.triple_extractor.model})..."
            )
            entities = []
            relations = []
            try:
                llm_entities, llm_relations = self.triple_extractor.extract_from_questions(
                    questions, show_progress=True
                )
                entities = llm_entities
                relations = llm_relations
                logger.info(
                    f"    LLM 抽取完成: {len(llm_entities)} 实体, "
                    f"{len(llm_relations)} 关系"
                )
            except Exception as e:
                logger.error(f"    LLM 三元组抽取失败: {e}", exc_info=True)
        else:
            logger.info("    LLM 三元组抽取已禁用，使用标题实体（无关系）")
            entities = structured_entities
            relations = []

        if not entities and not relations:
            logger.warning("没有实体和关系可用于构建知识图谱")
            return KnowledgeGraph()

        # Step 3: 构建图谱
        graph = self.graph_builder.build_graph(entities, relations)

        return graph

    # ==========================================
    # 便捷方法
    # ==========================================

    def load_experts(self, kb_path: Optional[str] = None) -> List[DomainExpert]:
        """
        从已构建的知识库加载所有专家

        Args:
            kb_path: 知识库根目录，None 则使用默认路径

        Returns:
            专家列表
        """
        kb_path = kb_path or self.kb_base_path

        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"知识库目录不存在: {kb_path}")

        experts = []

        # 查找所有专家目录
        for item in sorted(os.listdir(kb_path)):
            expert_dir = os.path.join(kb_path, item)
            metadata_file = os.path.join(expert_dir, "metadata.json")

            if os.path.isdir(expert_dir) and os.path.exists(metadata_file):
                try:
                    expert = DomainExpert.load_from_path(
                        expert_dir, config=self._config
                    )
                    experts.append(expert)
                except Exception as e:
                    logger.warning(f"加载专家 '{item}' 失败: {e}")

        logger.info(f"从 {kb_path} 加载了 {len(experts)} 个专家")
        return experts

    def get_build_info(self) -> Dict[str, Any]:
        """获取构建信息"""
        return dict(self._build_info)

    def __repr__(self) -> str:
        return (
            f"KBBuildPipeline(output='{self.kb_base_path}', "
            f"experts={len(self._experts)})"
        )
