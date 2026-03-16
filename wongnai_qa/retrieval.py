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
)


ARTIFACTS_DIR = VECTOR_DB_DIR.parent / "artifacts"
TUNED_WEIGHTS_PATH = ARTIFACTS_DIR / "tuned_retriever_weights.json"


@dataclass(frozen=True)
class ScoringWeights:
    rating: float = RATING_WEIGHT
    tag: float = TAG_MATCH_WEIGHT
    keyword: float = KEYWORD_MATCH_WEIGHT
    exact: float = EXACT_PHRASE_WEIGHT


class E5Embeddings(Embeddings):
    def __init__(self, model_name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
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
        matched_terms = sum(1 for term in query_terms if term in doc_text)
        score += (matched_terms / len(query_terms)) * weights.keyword

    if query_text and query_text in doc_text:
        score += weights.exact

    return score


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
    return _deduplicate_documents(ranked, k=k)


def retrieve_documents(
    vector_store: Chroma,
    query_profile: dict[str, Any],
    k: int = RETRIEVER_K,
    fetch_k: int = RETRIEVER_FETCH_K,
    weights: ScoringWeights | None = None,
) -> list[Document]:
    candidates = vector_store.similarity_search(query_profile["expanded_query"], k=fetch_k)
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda document: _score_document(document, query_profile, weights=weights),
        reverse=True,
    )
    return _deduplicate_documents(ranked, k=k)


def document_matches_query(document: Document, query_profile: dict[str, Any]) -> bool:
    metadata = document.metadata
    for group_name, values in query_profile["detected_tags"].items():
        if not values:
            continue
        doc_values = {value for value in str(metadata.get(group_name, "")).split("|") if value}
        if doc_values.intersection(values):
            return True

    doc_text = document.page_content.lower()
    doc_known_terms = {value for value in str(metadata.get("known_terms", "")).split("|") if value}
    for term in query_profile["query_terms"]:
        lowered_term = term.lower()
        if lowered_term in doc_text or lowered_term in doc_known_terms:
            return True
    return False
