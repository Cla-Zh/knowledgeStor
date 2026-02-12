"""
NSEQA - 神经-符号混合分布式知识专家问答系统

主入口程序，支持三种运行模式：
- build:  从数据集构建分布式专家知识库
- query:  回答单个问题
- eval:   在测试集上评估系统性能
- interactive: 交互式问答

Usage:
    # 构建知识库
    python main.py --mode build --dataset data/raw/2wikimultihopqa/train.json

    # 回答问题
    python main.py --mode query --question "Who is the CEO of Tesla?"

    # 评估系统
    python main.py --mode eval --dataset data/raw/2wikimultihopqa/test.json

    # 交互式问答
    python main.py --mode interactive
"""

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(level: str = "INFO"):
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_mode(args):
    """构建知识库"""
    from pipeline.kb_build_pipeline import KBBuildPipeline

    if not args.dataset:
        print("错误: --dataset 参数必填 (数据集路径)")
        sys.exit(1)

    pipeline = KBBuildPipeline(config_path=args.config)
    experts = pipeline.run(
        dataset_path=args.dataset,
        dataset_type=args.dataset_type,
        max_samples=args.max_samples,
    )

    print(f"\n知识库构建完成！共 {len(experts)} 个专家")
    for expert in experts:
        stats = expert.get_statistics()
        print(
            f"  {expert.expert_id}: {expert.domain.name} "
            f"(向量={stats['vector_index_size']}, "
            f"实体={stats['graph_entities']}, "
            f"关系={stats['graph_relations']})"
        )


def query_mode(args):
    """回答单个问题"""
    from pipeline.qa_pipeline import QAPipeline

    if not args.question:
        print("错误: --question 参数必填")
        sys.exit(1)

    qa = QAPipeline(kb_path=args.kb_path, config_path=args.config)
    answer = qa.answer(args.question)

    print(f"\nQuestion: {args.question}")
    print(f"Answer:   {answer.text}")
    print(f"Confidence: {answer.confidence:.2f}")
    print(f"Source:   {answer.source}")

    if answer.reasoning_path:
        print("\nReasoning Path:")
        for step in answer.reasoning_path:
            print(f"  {step}")

    if answer.explanation:
        print(f"\nExplanation:\n{answer.explanation}")


def eval_mode(args):
    """评估系统"""
    from pipeline.qa_pipeline import QAPipeline
    from evaluation.evaluator import Evaluator

    if not args.dataset:
        print("错误: --dataset 参数必填 (测试集路径)")
        sys.exit(1)

    qa = QAPipeline(kb_path=args.kb_path, config_path=args.config)
    evaluator = Evaluator(qa_pipeline=qa)

    save_path = args.output or "evaluation_results.json"
    results = evaluator.evaluate(
        test_data=args.dataset,
        dataset_type=args.dataset_type,
        max_samples=args.max_samples,
        save_path=save_path,
    )

    print(results.summary())


def interactive_mode(args):
    """交互式问答"""
    from pipeline.qa_pipeline import QAPipeline

    qa = QAPipeline(kb_path=args.kb_path, config_path=args.config)
    qa.interactive()


def main():
    parser = argparse.ArgumentParser(
        description="NSEQA - 神经-符号混合分布式知识专家问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode build --dataset data/raw/2wikimultihopqa/train.json
  python main.py --mode query --question "Who directed Inception?"
  python main.py --mode eval --dataset data/raw/2wikimultihopqa/test.json
  python main.py --mode interactive
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["build", "query", "eval", "interactive"],
        required=True,
        help="运行模式: build(构建知识库), query(单题问答), eval(评估), interactive(交互式)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集路径 (build/eval 模式必填)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["2wiki", "musique", "military"],
        default=None,
        help="数据集类型 (2wiki/musique/military)，默认自动检测",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="问题文本 (query 模式必填)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--kb-path",
        type=str,
        default="./data/knowledge_bases",
        help="知识库路径 (默认: ./data/knowledge_bases)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数 (用于调试)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (eval 模式)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)",
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(args.log_level)

    # 执行对应模式
    mode_handlers = {
        "build": build_mode,
        "query": query_mode,
        "eval": eval_mode,
        "interactive": interactive_mode,
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
