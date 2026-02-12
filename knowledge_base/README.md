# NSEQA - 神经-符号混合分布式知识专家问答系统

**NeuroSymbolic Expert QA** — 结合分布式知识专家系统与神经-符号混合架构，通过 OpenAI 兼容 API 调用外部模型，配合外部知识增强在多跳问答任务上实现高质量推理。

---

## 核心理念

| 维度 | 传统方法 | NSEQA |
|------|----------|-------|
| 知识组织 | 单一大知识库 | 分布式专家知识库，按领域自动划分 |
| 检索方式 | 纯向量检索 | 向量检索（神经） + 图谱查询（符号）混合 |
| 推理能力 | 单跳检索 | 多跳链式/图/迭代推理 |
| 可解释性 | 黑盒 | 完整推理路径可追溯 |

**三大支柱：**

1. **分布式知识专家** — 自动从数据集识别领域边界，每个专家只负责自己擅长的领域，通过智能路由协作解决复杂问题
2. **神经-符号混合** — 神经网络理解自然语言，符号系统执行精确推理和计算，两者互补
3. **多跳推理** — 支持链式推理、图推理、迭代精炼三种方案，处理复杂的多步问答

---

## 项目结构

```
NSEQA/
├── main.py                         # 主入口程序
├── config.yaml                     # 全局配置文件
├── requirements.txt                # Python 依赖
├── README.md                       # 本文件
│
├── data/                           # 数据层
│   ├── models.py                   # 核心数据模型 (Entity, Relation, Question, Fact...)
│   └── loader.py                   # 数据集加载器 (2WikiMultihopQA, MuSiQue)
│
├── kb_builder/                     # 知识库构建系统
│   ├── models.py                   # VectorIndex, KnowledgeGraph, ExpertDomain
│   ├── vector_builder.py           # 向量索引构建器 (FAISS / OpenAI Embedding API)
│   ├── graph_builder.py            # 知识图谱构建器 (NetworkX)
│   └── expert_identifier.py        # 专家领域自动识别 (聚类/LDA/规则)
│
├── experts/                        # 分布式专家系统
│   ├── base_expert.py              # 专家基类 (BaseExpert) + 默认实现 (DomainExpert)
│   └── expert_router.py            # 专家路由器 (语义相似度/关键词/混合)
│
├── neuro_symbolic/                 # 神经-符号转换层
│   └── semantic_parser.py          # 语义解析器 (实体提取/关系识别/查询分解)
│
├── qa_engine/                      # 问答推理引擎
│   ├── question_understanding.py   # 问题理解 (类型/复杂度/策略选择)
│   ├── multi_hop_reasoner.py       # 多跳推理器 (链式/图/迭代)
│   └── answer_generator.py         # 答案生成器 (符号/抽取/Chat API 生成式)
│
├── pipeline/                       # 流程编排
│   ├── kb_build_pipeline.py        # 知识库构建完整流程
│   └── qa_pipeline.py              # 问答完整流程
│
├── evaluation/                     # 评估模块
│   ├── metrics.py                  # 评估指标 (EM, F1, Accuracy)
│   └── evaluator.py                # 评估器 + 基线对比
│
└── utils/                          # 工具模块
    └── config.py                   # 配置管理器 (YAML 加载, 点号路径访问)
```

---

## 快速开始

### 1. 环境准备

```bash
# 建议使用 Python 3.9+
pip install -r requirements.txt

# (可选) 下载 SpaCy 英文模型，用于增强实体识别
python -m spacy download en_core_web_sm
```

### 1.5 配置 API 密钥

所有外部模型调用（Embedding 和 Chat 生成）均通过 OpenAI 兼容 API 完成，**无需本地 GPU 或模型下载**。

编辑 `config.yaml` 中的 `api` 部分，填入你的 API 信息：

```yaml
api:
  embedding:
    api_key: 'sk-your-embedding-key'
    base_url: 'https://api.openai.com/v1'
    model: 'text-embedding-3-small'
    dimension: 1536
  chat:
    api_key: 'sk-your-chat-key'
    base_url: 'https://api.openai.com/v1'
    model: 'gpt-4o-mini'
```

> **兼容说明**：任何兼容 OpenAI API 格式的服务均可使用，包括 Azure OpenAI、vLLM、Ollama、LiteLLM、DeepSeek 等。只需修改 `base_url` 和 `model` 即可。

### 2. 配置 API

编辑 `config.yaml`，填入你的 API 信息：

```yaml
api:
  embedding:
    api_key: 'sk-your-real-key'                    # 你的 Embedding API Key
    base_url: 'https://api.openai.com/v1'          # API 地址
    model: 'text-embedding-3-small'                # 模型名称
    dimension: 1536                                # 向量维度

  chat:
    api_key: 'sk-your-real-key'                    # 你的 Chat API Key
    base_url: 'https://api.openai.com/v1'          # API 地址
    model: 'gpt-4o-mini'                           # 模型名称
```

> **兼容性说明：** 支持任何 OpenAI 兼容的 API 服务（如 Azure OpenAI、DeepSeek、Ollama、vLLM、LiteLLM 等），只需修改 `base_url` 和 `model` 即可。Embedding 和 Chat 可以使用不同的 API 提供商。

### 3. 准备数据集

将数据集放到 `data/raw/` 目录下：

```
data/raw/
├── 2wikimultihopqa/
│   ├── train.json
│   ├── dev.json
│   └── test.json
└── musique/
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl
```

- **2WikiMultihopQA**: [下载地址](https://github.com/Alab-NII/2wikimultihop)
- **MuSiQue**: [下载地址](https://github.com/StonyBrookNLP/musique)

### 4. 构建知识库

```bash
python main.py --mode build --dataset data/raw/2wikimultihopqa/train.json
```

构建流程会自动：
1. 加载并解析数据集
2. 识别专家领域（人物/地理/组织等）
3. 将数据分配给各专家
4. 为每个专家构建向量索引（FAISS）和知识图谱（NetworkX）
5. 保存到 `data/knowledge_bases/`

**调试模式**（限制数据量加速测试）：

```bash
python main.py --mode build --dataset data/raw/2wikimultihopqa/train.json --max-samples 500
```

### 5. 问答

**单题问答：**

```bash
python main.py --mode query --question "Who is the director of the movie Inception?"
```

**交互式问答：**

```bash
python main.py --mode interactive
```

### 6. 评估

```bash
python main.py --mode eval \
    --dataset data/raw/2wikimultihopqa/test.json \
    --output results.json \
    --max-samples 200
```

评估指标包括：
- **Exact Match (EM)** — 精确匹配率
- **F1 Score** — Token 级 F1 分数
- **Answer Accuracy** — 答案准确率（支持别名匹配）
- 按问题类型和推理跳数分类统计

### 7. Python API 使用

```python
from pipeline.kb_build_pipeline import KBBuildPipeline
from pipeline.qa_pipeline import QAPipeline

# 构建知识库
builder = KBBuildPipeline(config_path="config.yaml")
builder.run("data/raw/2wikimultihopqa/train.json")

# 问答
qa = QAPipeline(kb_path="data/knowledge_bases")
answer = qa.answer("What is the capital of France?")

print(f"答案: {answer.text}")
print(f"置信度: {answer.confidence:.2f}")
print(f"推理路径:")
for step in answer.reasoning_path:
    print(f"  {step}")
```

---

## 配置文件说明

所有配置集中在 `config.yaml` 中，支持按需调整：

### system — 系统设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_experts` | `null` | 专家数量，`null` 表示自动确定（通过轮廓系数） |
| `expert_identification_method` | `clustering` | 领域识别算法：`clustering` / `lda` / `entity_based` |
| `random_seed` | `42` | 随机种子 |
| `log_level` | `INFO` | 日志级别 |

### api.embedding — Embedding API 设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `api_key` | `sk-xxx` | OpenAI 兼容 API Key |
| `base_url` | `https://api.openai.com/v1` | API 基础 URL（支持任何兼容接口） |
| `model` | `text-embedding-3-small` | Embedding 模型名称 |
| `dimension` | `1536` | 向量维度（需与模型匹配） |
| `batch_size` | `64` | 每次 API 调用的最大文本数 |
| `max_retries` | `3` | API 调用失败重试次数 |
| `timeout` | `60` | 请求超时时间（秒） |

### api.chat — Chat/生成模型 API 设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `api_key` | `sk-xxx` | OpenAI 兼容 API Key |
| `base_url` | `https://api.openai.com/v1` | API 基础 URL（支持任何兼容接口） |
| `model` | `gpt-4o-mini` | Chat 模型名称（用于答案生成） |
| `temperature` | `0.0` | 生成温度（0 = 确定性输出） |
| `max_tokens` | `512` | 最大生成 token 数 |
| `max_retries` | `3` | API 调用失败重试次数 |
| `timeout` | `60` | 请求超时时间（秒） |

### retrieval — 检索设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vector_top_k` | `5` | 向量检索返回的 top-k 结果数 |
| `rerank` | `true` | 是否启用重排序 |
| `hybrid_alpha` | `0.7` | 混合检索中向量权重（`1 - alpha` 为图谱权重） |
| `similarity_threshold` | `0.3` | 相似度阈值 |

### reasoning — 推理设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_hops` | `5` | 最大推理跳数 |
| `max_iterations` | `3` | 迭代精炼最大轮数 |
| `reasoning_mode` | `chain` | 推理模式：`chain`（链式） / `graph`（图） / `iterative`（迭代） |
| `confidence_threshold` | `0.5` | 置信度阈值 |

### storage — 存储路径

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kb_base_path` | `./data/knowledge_bases` | 知识库存储根路径 |
| `raw_data_path` | `./data/raw` | 原始数据集路径 |
| `processed_data_path` | `./data/processed` | 预处理数据路径 |
| `cache_enabled` | `true` | 是否启用缓存 |

### vector_index — 向量索引设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `index_type` | `faiss` | 索引类型：`faiss` / `simple`（numpy 回退） |
| `normalize_embeddings` | `true` | 是否归一化向量（余弦相似度） |

### graph — 知识图谱设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_path_length` | `3` | 路径查询最大跳数 |
| `min_confidence` | `0.5` | 关系最低置信度阈值 |

---

## 系统架构

```
                  用户问题 (自然语言)
                         │
           ┌─────────────▼─────────────┐
           │    问题理解 + 语义解析       │  ← 神经层：NER、关系识别、问题分解
           └─────────────┬─────────────┘
                         │
           ┌─────────────▼─────────────┐
           │      专家路由器             │  ← 将问题路由到最相关的专家
           └──┬──────────┬──────────┬──┘
              │          │          │
         ┌────▼───┐ ┌───▼────┐ ┌──▼─────┐
         │ 专家 A  │ │ 专家 B  │ │ 专家 C  │  ← 每个专家 = 向量索引 + 知识图谱
         │(人物)   │ │(地理)   │ │(组织)   │
         └────┬───┘ └───┬────┘ └──┬─────┘
              │         │         │
           ┌──▼─────────▼─────────▼──┐
           │    多跳推理引擎            │  ← 链式/图/迭代推理
           └─────────────┬───────────┘
                         │
           ┌─────────────▼─────────────┐
           │    答案生成器               │  ← 符号直接返回 / 抽取式 / 生成式
           └─────────────┬─────────────┘
                         │
                  最终答案 + 推理路径
```

---

## 支持的数据集

| 数据集 | 格式 | 特点 |
|--------|------|------|
| **2WikiMultihopQA** | JSON | 基于维基百科的多跳问答，含证据链和支撑事实 |
| **MuSiQue** | JSONL | 多步推理问答，含问题分解结构 |

系统通过 `DatasetLoader` 自动检测数据集格式，也可以通过 `--dataset-type` 手动指定。

---

## 命令行参数一览

```
python main.py --help
```

| 参数 | 说明 | 适用模式 |
|------|------|----------|
| `--mode` | 运行模式 (必填) | 全部 |
| `--dataset` | 数据集路径 | build, eval |
| `--dataset-type` | 数据集类型 (`2wiki` / `musique`) | build, eval |
| `--question` | 问题文本 | query |
| `--config` | 配置文件路径，默认 `config.yaml` | 全部 |
| `--kb-path` | 知识库路径，默认 `./data/knowledge_bases` | query, eval, interactive |
| `--max-samples` | 限制样本数（调试用） | build, eval |
| `--output` | 评估结果输出路径 | eval |
| `--log-level` | 日志级别 (`DEBUG`/`INFO`/`WARNING`/`ERROR`) | 全部 |

---

## 核心模块说明

### 数据层 (`data/`)

- **`models.py`** — 定义 `Entity`、`Relation`、`Fact`、`Question`、`Document`、`ReasoningChain` 等核心数据结构，全部支持 `to_dict()` / `from_dict()` 序列化
- **`loader.py`** — `DatasetLoader` 自动加载和解析数据集，提取实体、关系，构建文档列表

### 知识库构建 (`kb_builder/`)

- **`expert_identifier.py`** — 三种领域自动识别算法：
  - `clustering`: K-Means + TF-IDF，轮廓系数自动选 K
  - `lda`: LDA 主题模型
  - `entity_based`: 预定义实体类型规则
- **`vector_builder.py`** — 通过 OpenAI 兼容 Embedding API 将文本编码为向量并构建 FAISS 索引，自动批量处理与重试
- **`graph_builder.py`** — 从实体和关系构建 NetworkX 有向多重图，含实体去重、关系归一化、孤立节点移除

### 专家系统 (`experts/`)

- **`base_expert.py`** — 每个专家持有独立的向量索引和知识图谱，支持向量检索、符号检索（单跳/模式匹配/路径查询）和混合检索
- **`expert_router.py`** — 三种路由策略：语义相似度、关键词匹配、混合策略；支持多跳查询逐跳路由

### 推理引擎 (`qa_engine/`)

- **`question_understanding.py`** — 分析问题类型、复杂度（1-10 分）、估算跳数、选择推理策略
- **`multi_hop_reasoner.py`** — 三种推理方案：
  - **链式推理**: 逐跳检索 → 传递中间结果 → 生成下一跳查询
  - **图推理**: 将问题转为图查询模式，在知识图谱上直接路径搜索
  - **迭代精炼**: 初始检索 → 评估是否足够 → 补充查询 → 循环
- **`answer_generator.py`** — 三级答案生成：符号精确返回 → 上下文抽取 → Chat API 生成式

---

## 开发指南

### 添加新的数据集

在 `data/loader.py` 中新增加载方法：

```python
class DatasetLoader:
    def load_my_dataset(self, path: str) -> List[Question]:
        # 实现加载逻辑，返回 Question 列表
        ...
```

### 自定义专家类型

继承 `BaseExpert` 并实现抽象方法：

```python
from experts.base_expert import BaseExpert

class MySpecialExpert(BaseExpert):
    def retrieve_vector(self, query, top_k=5):
        # 自定义向量检索逻辑
        ...

    def retrieve_symbolic(self, query):
        # 自定义符号检索逻辑
        ...
```

### 添加新的推理策略

在 `qa_engine/multi_hop_reasoner.py` 的 `reason()` 方法中添加新的策略分支即可。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 向量编码 | OpenAI 兼容 Embedding API (`text-embedding-3-small` 等) |
| 答案生成 | OpenAI 兼容 Chat API (`gpt-4o-mini` 等) |
| 向量索引 | FAISS |
| 知识图谱 | NetworkX |
| 领域识别 | scikit-learn (K-Means, LDA) |
| 实体识别 | SpaCy + 规则引擎 |
| API 客户端 | openai (Python SDK) |
| 配置管理 | PyYAML |

---

## 许可证

本项目仅用于学术研究目的。
