from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from wongnai_qa.config import INDEX_SAMPLE_SIZE, QUERY_LABELS_ALGO_PATH, QUERY_LABELS_JUDGES_PATH
from wongnai_qa.preprocessing import analyze_query, load_resource_bundle, normalize_text
from wongnai_qa.retrieval import (
    ScoringWeights,
    document_matches_query,
    load_tuned_weights,
    retrieve_documents,
    retrieve_documents_baseline,
    save_tuned_weights,
)
from wongnai_qa.service import get_service


@dataclass(frozen=True)
class BenchmarkExample:
    query: str
    source: str
    query_profile: dict[str, Any]


def _read_query_lines(path, source: str, limit: int | None = None) -> list[BenchmarkExample]:
    bundle = load_resource_bundle()
    examples: list[BenchmarkExample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = normalize_text(raw_line)
            if not line:
                continue
            parts = [normalize_text(part) for part in line.split("|") if normalize_text(part)]
            query = normalize_text(" ".join(parts))
            if len(query) < 2:
                continue
            profile = analyze_query(query, resource_bundle=bundle)
            if not any(profile["detected_tags"].values()) and not profile["query_terms"]:
                continue
            examples.append(BenchmarkExample(query=query, source=source, query_profile=profile))
            if limit and len(examples) >= limit:
                break
    return examples


def build_benchmark(limit: int | None = 200) -> list[BenchmarkExample]:
    per_source_limit = limit if limit else None
    examples = _read_query_lines(QUERY_LABELS_ALGO_PATH, "algo", limit=per_source_limit)
    if not limit or len(examples) < limit:
        remaining = None if not limit else limit - len(examples)
        examples.extend(_read_query_lines(QUERY_LABELS_JUDGES_PATH, "judges", limit=remaining))

    deduped: list[BenchmarkExample] = []
    seen = set()
    for example in examples:
        key = example.query.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
        if limit and len(deduped) >= limit:
            break
    return deduped


def split_benchmark(
    examples: list[BenchmarkExample],
    train_ratio: float = 0.8,
) -> tuple[list[BenchmarkExample], list[BenchmarkExample]]:
    boundary = max(1, int(len(examples) * train_ratio))
    return examples[:boundary], examples[boundary:]


def _evaluate_queries(
    examples: list[BenchmarkExample],
    *,
    sample_size: int,
    top_k: int,
    fetch_k: int,
    mode: str,
    weights: ScoringWeights | None = None,
) -> dict[str, Any]:
    service = get_service(sample_size=sample_size)
    vector_store = service.ensure_vector_store(rebuild=False)

    hit_scores: list[float] = []
    relevant_counts: list[float] = []

    for example in examples:
        if mode == "baseline":
            docs = retrieve_documents_baseline(
                vector_store,
                query_profile=example.query_profile,
                k=top_k,
                fetch_k=fetch_k,
            )
        elif mode == "finetuned":
            docs = retrieve_documents(
                vector_store,
                query_profile=example.query_profile,
                k=top_k,
                fetch_k=fetch_k,
                weights=weights,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        matched = [document_matches_query(document, example.query_profile) for document in docs]
        relevant_count = sum(1 for item in matched if item)
        hit_scores.append(1.0 if relevant_count > 0 else 0.0)
        relevant_counts.append(relevant_count / max(1, len(docs)))

    return {
        "num_queries": len(examples),
        "hit_rate_at_k": round(mean(hit_scores), 4) if hit_scores else 0.0,
        "avg_relevant_ratio_at_k": round(mean(relevant_counts), 4) if relevant_counts else 0.0,
        "mode": mode,
        "weights": asdict(weights) if weights else None,
    }


def tune_retriever_weights(
    sample_size: int = INDEX_SAMPLE_SIZE,
    benchmark_limit: int = 64,
    top_k: int = 4,
    fetch_k: int = 6,
) -> dict[str, Any]:
    examples = build_benchmark(limit=benchmark_limit)
    train_examples, eval_examples = split_benchmark(examples)

    candidate_weights = [
        ScoringWeights(rating=rating, tag=tag, keyword=keyword, exact=exact)
        for rating in [0.05, 0.12]
        for tag in [0.45]
        for keyword in [0.33]
        for exact in [0.0]
    ]

    best_weights = None
    best_metrics = None
    best_score = -1.0

    for weights in candidate_weights:
        metrics = _evaluate_queries(
            train_examples,
            sample_size=sample_size,
            top_k=top_k,
            fetch_k=fetch_k,
            mode="finetuned",
            weights=weights,
        )
        score = metrics["hit_rate_at_k"] + metrics["avg_relevant_ratio_at_k"]
        if score > best_score:
            best_score = score
            best_weights = weights
            best_metrics = metrics

    assert best_weights is not None
    save_tuned_weights(best_weights)

    baseline_eval = _evaluate_queries(
        eval_examples,
        sample_size=sample_size,
        top_k=top_k,
        fetch_k=fetch_k,
        mode="baseline",
    )
    finetuned_eval = _evaluate_queries(
        eval_examples,
        sample_size=sample_size,
        top_k=top_k,
        fetch_k=fetch_k,
        mode="finetuned",
        weights=best_weights,
    )

    return {
        "train_queries": len(train_examples),
        "eval_queries": len(eval_examples),
        "best_train_metrics": best_metrics,
        "baseline_eval": baseline_eval,
        "finetuned_eval": finetuned_eval,
        "selected_weights": asdict(best_weights),
    }


def evaluate_current_models(
    sample_size: int = INDEX_SAMPLE_SIZE,
    benchmark_limit: int = 80,
    top_k: int = 4,
    fetch_k: int = 6,
) -> dict[str, Any]:
    examples = build_benchmark(limit=benchmark_limit)
    _, eval_examples = split_benchmark(examples)
    tuned_weights = load_tuned_weights()
    return {
        "benchmark_queries": len(eval_examples),
        "baseline": _evaluate_queries(
            eval_examples,
            sample_size=sample_size,
            top_k=top_k,
            fetch_k=fetch_k,
            mode="baseline",
        ),
        "finetuned": _evaluate_queries(
            eval_examples,
            sample_size=sample_size,
            top_k=top_k,
            fetch_k=fetch_k,
            mode="finetuned",
            weights=tuned_weights,
        ),
        "active_tuned_weights": asdict(tuned_weights),
    }
