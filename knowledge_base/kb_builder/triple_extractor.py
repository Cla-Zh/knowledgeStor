"""
LLM 驱动的三元组抽取器 (Triple Extractor)

通过 LLM（大语言模型）从原始文本中自动抽取实体-关系三元组 (Head, Relation, Tail)，
用于知识图谱构建。

参考 AutoSchemaKG 的 prompt 设计与 JSON 解析策略，支持：
1. 实体-关系三元组抽取（Head=名词, Relation=动词/关系, Tail=名词）
2. 文本分块处理（避免超过 LLM token 限制）
3. 批量并发调用
4. json_repair 容错解析 + DeepSeek <think> 标签处理

所有 LLM 调用均通过 OpenAI 兼容 Chat API 完成。
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import json_repair

from data.models import Entity, EntityType, Question, Relation
from utils.config import Config

logger = logging.getLogger(__name__)

# ============================================================
# Prompt 模板（直接参考 AutoSchemaKG zh-CN 版本）
# Head / Tail 明确标注为 "{名词}"，引导 LLM 输出短名词而非长句
# ============================================================

TRIPLE_EXTRACTION_SYSTEM_PROMPT = (
    "你是一个始终以有效JSON数组格式回应的助手"
)

TRIPLE_EXTRACTION_USER_PROMPT = (
    "给定一段文字，提取所有重要实体及其关系，并以简洁的方式总结。"
    "关系描述应清晰表达实体间的联系，且不重复头尾实体的信息。"
    "实体需具体明确，排除代词。\n"
    "返回格式必须为以下JSON结构,内容需用简体中文表述:\n"
    "[\n"
    '    {{\n'
    '        "Head": "{{名词}}",\n'
    '        "Relation": "{{动词或关系描述}}",\n'
    '        "Tail": "{{名词}}"\n'
    "    }}...\n"
    "]\n\n"
    "给定以下段落：\n{text}"
)

# 三元组所需字段（用于 key 修正）
REQUIRED_KEYS = {"Head", "Relation", "Tail"}

# 文本分块默认参数
MAX_CHUNK_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200


def _normalize_key(key: str) -> str:
    """AutoSchemaKG 风格的 key 归一化"""
    return key.strip().lower()


class TripleExtractor:
    """
    LLM 驱动的三元组抽取器

    从原始文本中自动抽取 (Head, Relation, Tail) 三元组，
    转换为 Entity 和 Relation 对象，用于知识图谱构建。
    """

    def __init__(self, config: Optional[Config] = None):
        if config is None:
            try:
                config = Config.get_instance()
            except Exception:
                config = Config()

        self._config = config

        # Chat API 参数
        self.api_key = config.get("api.chat.api_key", "sk-xxx")
        self.base_url = config.get("api.chat.base_url", "https://api.openai.com/v1")
        self.model = config.get("api.chat.model", "gpt-4o-mini")
        self.temperature = config.get("api.chat.temperature", 0.0)
        self.max_tokens = config.get("api.chat.max_tokens", 2048)
        self.max_retries = config.get("api.chat.max_retries", 3)
        self.timeout = config.get("api.chat.timeout", 60)

        # 三元组抽取专用配置
        self.max_workers = config.get("graph.triple_extraction.max_workers", 3)
        self.max_chunk_chars = config.get(
            "graph.triple_extraction.max_chunk_chars", MAX_CHUNK_CHARS
        )
        self.chunk_overlap = config.get(
            "graph.triple_extraction.chunk_overlap", CHUNK_OVERLAP_CHARS
        )
        self.enabled = config.get("graph.triple_extraction.enabled", True)

        # 检测是否为 DeepSeek 模型
        self._is_deepseek = "deepseek" in self.base_url.lower() or "deepseek" in self.model.lower()

        # OpenAI 客户端（延迟初始化）
        self._client = None

        logger.info(
            f"TripleExtractor 初始化: model={self.model}, "
            f"max_workers={self.max_workers}, enabled={self.enabled}, "
            f"is_deepseek={self._is_deepseek}"
        )

    # ==========================================
    # API 客户端
    # ==========================================

    def _get_client(self):
        """获取 OpenAI Chat 客户端（延迟初始化）"""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            logger.info(f"TripleExtractor 客户端已初始化: {self.base_url}")

        return self._client

    # ==========================================
    # 文本分块
    # ==========================================

    def _split_text(self, text: str) -> List[str]:
        """
        将长文本分割为适合 LLM 处理的块。
        按段落边界切分，避免截断句子。
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        if len(text) <= self.max_chunk_chars:
            return [text]

        # 按段落分割（支持 \n 和 \r\n）
        paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 <= self.max_chunk_chars:
                current_chunk = (current_chunk + "\n" + para).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # 如果单个段落超过最大长度，强制切分
                if len(para) > self.max_chunk_chars:
                    step = max(self.max_chunk_chars - self.chunk_overlap, 100)
                    for i in range(0, len(para), step):
                        chunks.append(para[i : i + self.max_chunk_chars])
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # ==========================================
    # LLM 调用
    # ==========================================

    def _call_llm(self, text: str) -> List[Dict[str, str]]:
        """
        调用 LLM 从单个文本块中抽取三元组。

        参考 AutoSchemaKG:
        - 对 DeepSeek 使用 extra_body={"enable_thinking": False} 禁用思考模式
        - 使用 json_repair 做容错 JSON 解析
        """
        client = self._get_client()

        user_prompt = TRIPLE_EXTRACTION_USER_PROMPT.format(text=text)

        messages = [
            {"role": "system", "content": TRIPLE_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # 构建 API 调用参数
            api_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            # DeepSeek 专用: 禁用思考模式（参考 AutoSchemaKG）
            if self._is_deepseek:
                api_kwargs["extra_body"] = {"enable_thinking": False}

            response = client.chat.completions.create(**api_kwargs)

            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM 返回内容为空")
                return []

            content = content.strip()
            logger.debug(f"LLM 原始返回 (前200字): {content[:200]}")

            # 解析 JSON 响应
            triples = self._parse_llm_response(content)
            if triples:
                logger.debug(f"成功抽取 {len(triples)} 个三元组")
            return triples

        except Exception as e:
            logger.error(f"LLM 三元组抽取调用失败: {e}")
            return []

    def _parse_llm_response(self, content: str) -> List[Dict[str, str]]:
        """
        解析 LLM 返回的 JSON 内容。

        参考 AutoSchemaKG 的 fix_triple_extraction_response:
        1. 去除 <think>...</think> 标签
        2. 用 json_repair.loads 做容错解析
        3. 对 key 做归一化映射（head → Head 等）
        4. 去重
        """
        if not content:
            return []

        # Step 1: 去除 DeepSeek 思考标签
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        # Step 2: 找到 JSON 数组起始位置
        json_start = content.find("[")
        if json_start == -1:
            # 尝试补全
            content = "[" + content.strip() + "]"
        else:
            # 从 [ 开始到最后一个 ]
            json_end = content.rfind("]")
            if json_end > json_start:
                content = content[json_start : json_end + 1]
            else:
                content = content[json_start:]

        # Step 3: 使用 json_repair 容错解析（参考 AutoSchemaKG）
        try:
            parsed_objects = json_repair.loads(content)
        except Exception as e:
            logger.warning(f"json_repair 解析失败: {e}\n内容前300字: {content[:300]}")
            return []

        if not isinstance(parsed_objects, list):
            logger.warning(f"解析结果不是数组: type={type(parsed_objects).__name__}")
            return []

        if len(parsed_objects) == 0:
            return []

        # Step 4: Key 归一化 + 验证（参考 AutoSchemaKG 的 fix_triple_extraction_response）
        corrected_data: List[Dict[str, str]] = []
        seen_triples: set = set()

        for idx, item in enumerate(parsed_objects):
            if not isinstance(item, dict):
                continue

            # 归一化 key 到标准格式 (Head, Relation, Tail)
            corrected_item: Dict[str, str] = {}
            for key, value in item.items():
                norm_key = _normalize_key(key)
                # 寻找匹配的标准 key
                matching = [
                    exp_key
                    for exp_key in REQUIRED_KEYS
                    if _normalize_key(exp_key) in norm_key
                ]
                if len(matching) == 1:
                    corrected_item[matching[0]] = str(value).strip()
                else:
                    corrected_item[key] = str(value).strip()

            # 检查必需字段
            missing = REQUIRED_KEYS - corrected_item.keys()
            if missing:
                continue

            head = corrected_item["Head"]
            relation = corrected_item["Relation"]
            tail = corrected_item["Tail"]

            # 验证非空
            if not head or not relation or not tail:
                continue

            # 去重
            triple_key = (head, relation, tail)
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)

            corrected_data.append({
                "Head": head,
                "Relation": relation,
                "Tail": tail,
            })

        return corrected_data

    # ==========================================
    # 批量抽取
    # ==========================================

    def extract_from_text(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """从单篇文本中抽取实体和关系，自动分块处理长文本。"""
        chunks = self._split_text(text)
        if not chunks:
            return [], []

        all_triples = []
        for chunk in chunks:
            triples = self._call_llm(chunk)
            all_triples.extend(triples)

        return self._triples_to_entities_and_relations(all_triples)

    def extract_from_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        从多篇文本中批量抽取实体和关系。

        全部文本块使用 ThreadPoolExecutor 并发调用 LLM，
        并发数由 config.yaml 的 graph.triple_extraction.max_workers 控制。
        """
        if not texts:
            return [], []

        # 所有文本分块
        all_chunks: List[str] = []
        for text in texts:
            chunks = self._split_text(text)
            all_chunks.extend(chunks)

        if not all_chunks:
            return [], []

        total = len(all_chunks)
        logger.info(
            f"三元组抽取: {len(texts)} 篇文本 → {total} 个文本块, "
            f"并发线程={self.max_workers}, model={self.model}"
        )

        # 设置进度条
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total, desc="LLM 三元组抽取")
            except ImportError:
                pass

        all_triples: List[Dict[str, str]] = []
        success_count = 0
        fail_count = 0
        first_example_logged = False

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 一次性提交全部文本块
            future_to_idx = {
                executor.submit(self._call_llm, chunk): idx
                for idx, chunk in enumerate(all_chunks)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    triples = future.result(timeout=self.timeout + 30)
                    all_triples.extend(triples)
                    if triples:
                        success_count += 1
                        # 打印第一个成功的示例三元组
                        if not first_example_logged:
                            logger.info(f"  示例三元组: {triples[0]}")
                            first_example_logged = True
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.warning(f"文本块 {idx} 三元组抽取异常: {e}")
                    fail_count += 1

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        logger.info(
            f"三元组抽取完成: 共提取 {len(all_triples)} 个三元组, "
            f"成功={success_count}, 空结果={fail_count}"
        )

        return self._triples_to_entities_and_relations(all_triples)

    def extract_from_questions(
        self,
        questions: List[Question],
        show_progress: bool = True,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        从 Question 列表中抽取实体和关系。
        提取每个 Question 的上下文文本，交给 LLM 抽取三元组。
        """
        if not self.enabled:
            logger.info("LLM 三元组抽取已禁用 (graph.triple_extraction.enabled=false)")
            return [], []

        # 收集所有文本
        texts: List[str] = []
        for question in questions:
            # 优先使用完整文本（MilitaryData 文章）
            if question.text and len(question.text) > 100:
                texts.append(question.text)
            else:
                # 拼接上下文
                for title, sentences in question.context:
                    full_text = (
                        " ".join(sentences)
                        if isinstance(sentences, list)
                        else str(sentences)
                    )
                    if title:
                        full_text = f"{title}: {full_text}"
                    if full_text.strip():
                        texts.append(full_text)

        if not texts:
            logger.warning("没有可用于三元组抽取的文本")
            return [], []

        logger.info(f"准备从 {len(texts)} 篇文本中进行 LLM 三元组抽取...")
        return self.extract_from_texts(texts, show_progress)

    # ==========================================
    # 三元组 → Entity + Relation 转换
    # ==========================================

    def _triples_to_entities_and_relations(
        self,
        triples: List[Dict[str, str]],
    ) -> Tuple[List[Entity], List[Relation]]:
        """将三元组转换为 Entity 和 Relation 对象，自动去重实体。"""
        entities: List[Entity] = []
        relations: List[Relation] = []
        seen_entity_names: set = set()

        for triple in triples:
            head = triple["Head"]
            relation_text = triple["Relation"]
            tail = triple["Tail"]

            # 添加实体（去重）
            for name in (head, tail):
                if name not in seen_entity_names:
                    entities.append(
                        Entity(
                            name=name,
                            type=EntityType.OTHER.value,
                            source="llm_extraction",
                        )
                    )
                    seen_entity_names.add(name)

            # 添加关系
            relations.append(
                Relation(
                    subject=head,
                    predicate=relation_text,
                    object=tail,
                    confidence=0.8,
                    source="llm_extraction",
                )
            )

        logger.info(
            f"三元组转换完成: {len(entities)} 个实体, {len(relations)} 条关系"
        )
        return entities, relations

    def __repr__(self) -> str:
        return (
            f"TripleExtractor(model='{self.model}', "
            f"enabled={self.enabled})"
        )
