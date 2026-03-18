from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from wongnai_qa.config import DEFAULT_QUERY_SET, INDEX_SAMPLE_SIZE, RETRIEVER_FETCH_K, RETRIEVER_K
from wongnai_qa.generation import build_baseline_answer, build_rag_answer
from wongnai_qa.llm import load_llm
from wongnai_qa.preprocessing import analyze_query, load_and_preprocess_data, load_resource_bundle
from wongnai_qa.retrieval import (
    get_vector_store,
    retrieve_documents,
    retrieve_documents_baseline,
    vector_store_is_current,
)


@dataclass
class QueryResponse:
    query: str
    query_profile: dict[str, Any]
    baseline_answer: str
    finetuned_answer: str
    baseline_improved_answer: str | None
    finetuned_improved_answer: str | None
    baseline_retrieved_documents: list[dict[str, Any]]
    retrieved_documents: list[dict[str, Any]]


class WongnaiQAService:
    def __init__(self, sample_size: int = INDEX_SAMPLE_SIZE):
        self.sample_size = sample_size
        self._resource_bundle = load_resource_bundle()
        self._vector_store = None
        self._llm = None
        self._lock = Lock()

    def ensure_vector_store(self, rebuild: bool = False):
        with self._lock:
            if rebuild:
                documents = load_and_preprocess_data(sample_size=self.sample_size)
                self._vector_store = get_vector_store(
                    documents=documents,
                    sample_size=self.sample_size,
                )
            elif self._vector_store is None:
                if vector_store_is_current(self.sample_size):
                    self._vector_store = get_vector_store(
                        documents=None,
                        sample_size=self.sample_size,
                    )
                else:
                    documents = load_and_preprocess_data(sample_size=self.sample_size)
                    self._vector_store = get_vector_store(
                        documents=documents,
                        sample_size=self.sample_size,
                    )
            elif not vector_store_is_current(self.sample_size):
                documents = load_and_preprocess_data(sample_size=self.sample_size)
                self._vector_store = get_vector_store(
                    documents=documents,
                    sample_size=self.sample_size,
                )
        return self._vector_store

    def ensure_llm(self):
        with self._lock:
            if self._llm is None:
                self._llm = load_llm()
        return self._llm

    def query(
        self,
        question: str,
        top_k: int = RETRIEVER_K,
        fetch_k: int = RETRIEVER_FETCH_K,
        include_improved: bool = True,
        improved_mode: str = "both",
        rebuild: bool = False,
    ) -> QueryResponse:
        vector_store = self.ensure_vector_store(rebuild=rebuild)
        query_profile = analyze_query(question, resource_bundle=self._resource_bundle)
        baseline_documents = retrieve_documents_baseline(
            vector_store,
            query_profile=query_profile,
            k=top_k,
            fetch_k=fetch_k,
        )
        finetuned_documents = retrieve_documents(
            vector_store,
            query_profile=query_profile,
            k=top_k,
            fetch_k=fetch_k,
        )

        baseline_answer = build_baseline_answer(question, baseline_documents)
        finetuned_answer = build_baseline_answer(question, finetuned_documents).replace(
            "Baseline answer",
            "Finetuned retrieval answer",
            1,
        ).replace(
            "Baseline:",
            "Finetuned retrieval:",
            1,
        )
        baseline_improved_answer = None
        finetuned_improved_answer = None
        if include_improved:
            if improved_mode not in {"both", "baseline", "finetuned"}:
                raise ValueError(f"Unsupported improved_mode: {improved_mode}")
            llm = self.ensure_llm()
            if improved_mode in {"both", "baseline"}:
                baseline_improved_answer = build_rag_answer(question, baseline_documents, llm).replace(
                    "Improved answer",
                    "Baseline improved answer",
                    1,
                )
            if improved_mode in {"both", "finetuned"}:
                finetuned_improved_answer = build_rag_answer(question, finetuned_documents, llm).replace(
                    "Improved answer",
                    "Finetuned improved answer",
                    1,
                )

        serialized_baseline_documents = [
            {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            for document in baseline_documents
        ]
        serialized_documents = [
            {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            for document in finetuned_documents
        ]

        return QueryResponse(
            query=question,
            query_profile=query_profile,
            baseline_answer=baseline_answer,
            finetuned_answer=finetuned_answer,
            baseline_improved_answer=baseline_improved_answer,
            finetuned_improved_answer=finetuned_improved_answer,
            baseline_retrieved_documents=serialized_baseline_documents,
            retrieved_documents=serialized_documents,
        )

    def demo_queries(self) -> list[str]:
        return DEFAULT_QUERY_SET


_service: WongnaiQAService | None = None
_service_lock = Lock()


def get_service(sample_size: int = INDEX_SAMPLE_SIZE) -> WongnaiQAService:
    global _service
    with _service_lock:
        if _service is None or _service.sample_size != sample_size:
            _service = WongnaiQAService(sample_size=sample_size)
    return _service
