import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from wongnai_qa.config import (
    EMBEDDING_MODEL_NAME,
    EXACT_PHRASE_WEIGHT,
    HF_LOCAL_FILES_ONLY,
    INDEX_SAMPLE_SIZE,
    INDEX_VERSION,
    KEYWORD_MATCH_WEIGHT,
    RATING_WEIGHT,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    TAG_MATCH_WEIGHT,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_DIR,
    VECTOR_DB_META_PATH,
    resolve_cached_model_path,
)


ARTIFACTS_DIR = VECTOR_DB_DIR.parent / "artifacts"
TUNED_WEIGHTS_PATH = ARTIFACTS_DIR / "tuned_retriever_weights.json"

PRICE_CONTRADICTIONS = {
    "budget": {"expensive"},
    "expensive": {"budget"},
}

TEXT_CONTRADICTION_HINTS = {
    ("price", "budget"): ["แพง", "ราคาแรง", "แพงมาก", "ค่อนข้างแพง"],
    ("price", "expensive"): ["ถูก", "ไม่แพง", "ย่อมเยา", "ประหยัด", "คุ้ม"],
    ("ambience", "quiet"): ["คนเยอะ", "แน่น", "เสียงดัง", "วุ่นวาย", "พลุกพล่าน"],
}
STRICT_MATCH_GROUPS = ("cuisine", "food_type", "location")


@dataclass(frozen=True)
class ScoringWeights:
    rating: float = RATING_WEIGHT
    tag: float = TAG_MATCH_WEIGHT
    keyword: float = KEYWORD_MATCH_WEIGHT
    exact: float = EXACT_PHRASE_WEIGHT


class E5Embeddings(Embeddings):
    def __init__(self, model_name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = resolve_cached_model_path(model_name)
        self.base_embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device, "local_files_only": HF_LOCAL_FILES_ONLY},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base_embeddings.embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base_embeddings.embed_query(f"query: {text}")


def _index_metadata(sample_size: int | None) -> dict[str, Any]:
    return {
        "index_version": INDEX_VERSION,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "sample_size": sample_size,
        "collection_name": VECTOR_COLLECTION_NAME,
    }


def _load_index_metadata() -> dict[str, Any] | None:
    if not VECTOR_DB_META_PATH.exists():
        return None
    return json.loads(VECTOR_DB_META_PATH.read_text(encoding="utf-8"))


def vector_store_is_current(sample_size: int | None = INDEX_SAMPLE_SIZE) -> bool:
    if not VECTOR_DB_DIR.exists():
        return False
    current = _load_index_metadata()
    return current == _index_metadata(sample_size)


def _persist_index_metadata(sample_size: int | None) -> None:
    Path(VECTOR_DB_DIR).mkdir(parents=True, exist_ok=True)
    VECTOR_DB_META_PATH.write_text(
        json.dumps(_index_metadata(sample_size), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_embeddings() -> Embeddings:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return E5Embeddings(model_name=EMBEDDING_MODEL_NAME)


def load_tuned_weights() -> ScoringWeights:
    if not TUNED_WEIGHTS_PATH.exists():
        return ScoringWeights()
    payload = json.loads(TUNED_WEIGHTS_PATH.read_text(encoding="utf-8"))
    return ScoringWeights(
        rating=float(payload.get("rating", RATING_WEIGHT)),
        tag=float(payload.get("tag", TAG_MATCH_WEIGHT)),
        keyword=float(payload.get("keyword", KEYWORD_MATCH_WEIGHT)),
        exact=float(payload.get("exact", EXACT_PHRASE_WEIGHT)),
    )


def save_tuned_weights(weights: ScoringWeights) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TUNED_WEIGHTS_PATH.write_text(
        json.dumps(asdict(weights), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_vector_store(
    documents: list[Document] | None = None,
    sample_size: int | None = INDEX_SAMPLE_SIZE,
) -> Chroma:
    embeddings = get_embeddings()
    if documents is None and vector_store_is_current(sample_size):
        print("Loading existing Chroma index...")
        return Chroma(
            collection_name=VECTOR_COLLECTION_NAME,
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=embeddings,
        )

    if documents is None:
        raise ValueError("Documents are required to build or rebuild the vector store.")

    print("Building Chroma index from preprocessed documents...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=VECTOR_COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR),
    )
    _persist_index_metadata(sample_size)
    print("Chroma index built successfully.")
    return vector_store


def _score_document(
    document: Document,
    query_profile: dict[str, Any],
    weights: ScoringWeights | None = None,
) -> float:
    weights = weights or load_tuned_weights()
    query_text = query_profile["normalized_query"].lower()
    query_terms = set(term.lower() for term in query_profile["query_terms"])
    query_tokens = set(token.lower() for token in query_profile.get("query_tokens", []))
    doc_text = document.page_content.lower()
    metadata = document.metadata

    rating = float(metadata.get("rating", 0.0)) / 5.0
    score = rating * weights.rating

    tag_matches = 0
    tag_total = 0
    for group_name, values in query_profile["detected_tags"].items():
        if not values:
            continue
        tag_total += len(values)
        raw_value = metadata.get(group_name, "")
        doc_values = {value for value in str(raw_value).split("|") if value}
        tag_matches += len(doc_values.intersection(values))
    if tag_total:
        score += (tag_matches / tag_total) * weights.tag

    if query_terms:
        doc_known_terms = {value for value in str(metadata.get("known_terms", "")).split("|") if value}
        matched_terms = sum(
            1 for term in query_terms if term in doc_text or term in doc_known_terms
        )
        score += (matched_terms / len(query_terms)) * weights.keyword

    if query_tokens:
        matched_tokens = sum(1 for token in query_tokens if token in doc_text)
        score += (matched_tokens / len(query_tokens)) * (weights.keyword * 0.5)

    if query_text and query_text in doc_text:
        score += weights.exact

    score -= _conflict_penalty(document, query_profile)
    return score


def _conflict_penalty(document: Document, query_profile: dict[str, Any]) -> float:
    penalty = 0.0
    metadata = document.metadata
    doc_text = document.page_content.lower()

    for group_name, values in query_profile["detected_tags"].items():
        if not values:
            continue
        doc_values = {value for value in str(metadata.get(group_name, "")).split("|") if value}
        for value in values:
            conflicting_values = PRICE_CONTRADICTIONS.get(value, set())
            if conflicting_values.intersection(doc_values):
                penalty += 0.55

            contradiction_hints = TEXT_CONTRADICTION_HINTS.get((group_name, value), [])
            if any(hint in doc_text for hint in contradiction_hints):
                penalty += 0.35

        if group_name == "location" and doc_values and not doc_values.intersection(values):
            penalty += 0.25

    return penalty


def _match_stats(document: Document, query_profile: dict[str, Any]) -> dict[str, float]:
    metadata = document.metadata
    doc_text = document.page_content.lower()
    doc_known_terms = {value for value in str(metadata.get("known_terms", "")).split("|") if value}

    requested_groups = 0
    matched_groups = 0
    for group_name, values in query_profile["detected_tags"].items():
        if not values:
            continue
        requested_groups += 1
        doc_values = {value for value in str(metadata.get(group_name, "")).split("|") if value}
        if doc_values.intersection(values):
            matched_groups += 1

    query_terms = [term.lower() for term in query_profile.get("query_terms", [])]
    query_tokens = [token.lower() for token in query_profile.get("query_tokens", [])]
    matched_terms = sum(1 for term in query_terms if term in doc_text or term in doc_known_terms)
    matched_tokens = sum(1 for token in query_tokens if token in doc_text)

    group_ratio = matched_groups / requested_groups if requested_groups else 0.0
    term_ratio = matched_terms / len(query_terms) if query_terms else 0.0
    token_ratio = matched_tokens / len(query_tokens) if query_tokens else 0.0
    coverage_components = [
        ratio
        for ratio in (group_ratio, term_ratio, token_ratio)
        if ratio > 0
    ]
    coverage = sum(coverage_components) / len(coverage_components) if coverage_components else 0.0
    return {
        "group_ratio": group_ratio,
        "term_ratio": term_ratio,
        "token_ratio": token_ratio,
        "coverage": coverage,
    }


def _prioritize_matching_documents(
    documents: list[Document],
    query_profile: dict[str, Any],
    k: int,
) -> list[Document]:
    has_strict_requirements = any(query_profile["detected_tags"].get(group) for group in STRICT_MATCH_GROUPS)
    requested_locations = set(query_profile["detected_tags"].get("location", []))
    matched: list[Document] = []
    strict_only: list[Document] = []
    fallback: list[Document] = []
    for document in documents:
        strict_ok = _passes_strict_constraints(document, query_profile)
        if strict_ok and document_matches_query(document, query_profile):
            matched.append(document)
        elif strict_ok:
            strict_only.append(document)
        else:
            fallback.append(document)
    if has_strict_requirements:
        strict_ranked = _deduplicate_documents(matched + strict_only, k=k)
        if strict_ranked:
            return strict_ranked
        # If strict filtering found nothing, try candidates that still overlap
        # with at least one strict group (cuisine/food_type/location).
        partial_fallback = [
            document
            for document in fallback
            if _strict_overlap_score(document, query_profile) > 0
            and (not requested_locations or _has_location_overlap(document, requested_locations))
        ]
        if partial_fallback:
            return _deduplicate_documents(partial_fallback, k=k)
        if requested_locations:
            return []
        return _deduplicate_documents(fallback, k=k)
    return _deduplicate_documents(matched + strict_only + fallback, k=k)


def _passes_strict_constraints(document: Document, query_profile: dict[str, Any]) -> bool:
    metadata = document.metadata
    for group_name in STRICT_MATCH_GROUPS:
        requested_values = set(query_profile["detected_tags"].get(group_name, []))
        if not requested_values:
            continue
        raw_value = metadata.get(group_name, "")
        doc_values = {value for value in str(raw_value).split("|") if value}
        if group_name == "location" and not doc_values:
            return False
        if doc_values and not doc_values.intersection(requested_values):
            return False
    return True


def _strict_overlap_score(document: Document, query_profile: dict[str, Any]) -> int:
    metadata = document.metadata
    score = 0
    for group_name in STRICT_MATCH_GROUPS:
        requested_values = set(query_profile["detected_tags"].get(group_name, []))
        if not requested_values:
            continue
        raw_value = metadata.get(group_name, "")
        doc_values = {value for value in str(raw_value).split("|") if value}
        if doc_values.intersection(requested_values):
            score += 1
    return score


def _has_location_overlap(document: Document, requested_locations: set[str]) -> bool:
    raw_value = document.metadata.get("location", "")
    doc_values = {value for value in str(raw_value).split("|") if value}
    return bool(doc_values.intersection(requested_locations))


def _deduplicate_documents(documents: list[Document], k: int) -> list[Document]:
    unique_documents: list[Document] = []
    seen = set()
    for document in documents:
        key = (document.metadata.get("review_id"), document.metadata.get("chunk_id"))
        if key in seen:
            continue
        unique_documents.append(document)
        seen.add(key)
        if len(unique_documents) >= k:
            break
    return unique_documents


def retrieve_documents_baseline(
    vector_store: Chroma,
    query_profile: dict[str, Any],
    k: int = RETRIEVER_K,
    fetch_k: int = RETRIEVER_FETCH_K,
) -> list[Document]:
    candidates = vector_store.similarity_search(query_profile["normalized_query"], k=fetch_k)
    if not candidates:
        return []
    return _deduplicate_documents(candidates, k=k)


def rank_documents_by_profile(
    documents: list[Document],
    query_profile: dict[str, Any],
    k: int = RETRIEVER_K,
    weights: ScoringWeights | None = None,
) -> list[Document]:
    ranked = sorted(
        documents,
        key=lambda document: _score_document(document, query_profile, weights=weights),
        reverse=True,
    )
    return _prioritize_matching_documents(ranked, query_profile=query_profile, k=k)


def retrieve_documents(
    vector_store: Chroma,
    query_profile: dict[str, Any],
    k: int = RETRIEVER_K,
    fetch_k: int = RETRIEVER_FETCH_K,
    weights: ScoringWeights | None = None,
) -> list[Document]:
    strict_groups = sum(1 for group in STRICT_MATCH_GROUPS if query_profile["detected_tags"].get(group))
    if strict_groups >= 2:
        effective_fetch_k = max(fetch_k, k * 16)
    elif strict_groups == 1:
        effective_fetch_k = max(fetch_k, k * 10)
    else:
        effective_fetch_k = fetch_k

    candidates = vector_store.similarity_search(query_profile["expanded_query"], k=effective_fetch_k)
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda document: _score_document(document, query_profile, weights=weights),
        reverse=True,
    )
    return _prioritize_matching_documents(ranked, query_profile=query_profile, k=k)


def document_matches_query(document: Document, query_profile: dict[str, Any]) -> bool:
    stats = _match_stats(document, query_profile)
    requested_groups = sum(1 for values in query_profile["detected_tags"].values() if values)
    has_specific_terms = bool(query_profile.get("query_terms") or query_profile.get("query_tokens"))

    if requested_groups >= 2:
        return stats["group_ratio"] >= 0.5 or stats["coverage"] >= 0.55
    if requested_groups == 1:
        return stats["group_ratio"] >= 1.0 or stats["coverage"] >= 0.45
    if has_specific_terms:
        return max(stats["term_ratio"], stats["token_ratio"]) >= 0.34
    return False
