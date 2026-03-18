import argparse

from wongnai_qa.api import app
from wongnai_qa.config import INDEX_SAMPLE_SIZE, RETRIEVER_FETCH_K, RETRIEVER_K
from wongnai_qa.service import get_service


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wongnai restaurant QA demo")
    parser.add_argument("--query", action="append", help="Query to run. Can be used multiple times.")
    parser.add_argument("--sample-size", type=int, default=INDEX_SAMPLE_SIZE)
    parser.add_argument("--top-k", type=int, default=RETRIEVER_K)
    parser.add_argument("--fetch-k", type=int, default=RETRIEVER_FETCH_K)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the vector index.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip the improved generative answer.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    service = get_service(sample_size=args.sample_size)
    queries = args.query or service.demo_queries()

    print("=== Wongnai Restaurant QA System ===")
    print(f"Loaded queries: {len(queries)}")
    print(f"Top-k retrieval: {args.top_k}")
    print("")

    for query in queries:
        result = service.query(
            query,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            include_improved=not args.skip_llm,
            rebuild=args.rebuild,
        )
        print("=" * 80)
        print(f"Query: {result.query}")
        print(f"Detected tags: {result.query_profile['detected_tags']}")
        if result.query_profile["query_terms"]:
            print(f"Matched lexicon terms: {result.query_profile['query_terms']}")
        print("-" * 80)
        print("Baseline model")
        print(result.baseline_answer)
        print("-" * 80)
        print("Finetuned retriever")
        print(result.finetuned_answer)
        print("-" * 80)
        if (
            result.baseline_improved_answer is None
            or result.finetuned_improved_answer is None
        ):
            print("Generative answer: skipped (--skip-llm)")
        else:
            print("Generative answer from baseline retrieval")
            print(result.baseline_improved_answer)
            print("-" * 80)
            print("Generative answer from finetuned retrieval")
            print(result.finetuned_improved_answer)
        print("")
