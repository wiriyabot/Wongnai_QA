from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from wongnai_qa.config import INDEX_SAMPLE_SIZE, RETRIEVER_FETCH_K, RETRIEVER_K
from wongnai_qa.service import get_service


app = FastAPI(
    title="Wongnai Restaurant QA API",
    version="1.0.0",
    description="FastAPI backend for Wongnai restaurant retrieval and QA",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    sample_size: int = Field(default=INDEX_SAMPLE_SIZE, ge=1)
    top_k: int = Field(default=RETRIEVER_K, ge=1, le=10)
    fetch_k: int = Field(default=RETRIEVER_FETCH_K, ge=1, le=30)
    include_improved: bool = True
    improved_mode: str = Field(default="both", pattern="^(both|baseline|finetuned)$")
    rebuild: bool = False


class QueryResponseModel(BaseModel):
    query: str
    query_profile: dict
    baseline_answer: str
    finetuned_answer: str
    baseline_improved_answer: str | None
    finetuned_improved_answer: str | None
    baseline_retrieved_documents: list[dict]
    retrieved_documents: list[dict]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo-queries")
def demo_queries() -> dict[str, list[str]]:
    service = get_service()
    return {"queries": service.demo_queries()}


@app.post("/query", response_model=QueryResponseModel)
def query(payload: QueryRequest) -> QueryResponseModel:
    service = get_service(sample_size=payload.sample_size)
    result = service.query(
        payload.query,
        top_k=payload.top_k,
        fetch_k=payload.fetch_k,
        include_improved=payload.include_improved,
        improved_mode=payload.improved_mode,
        rebuild=payload.rebuild,
    )
    return QueryResponseModel(
        query=result.query,
        query_profile=result.query_profile,
        baseline_answer=result.baseline_answer,
        finetuned_answer=result.finetuned_answer,
        baseline_improved_answer=result.baseline_improved_answer,
        finetuned_improved_answer=result.finetuned_improved_answer,
        baseline_retrieved_documents=result.baseline_retrieved_documents,
        retrieved_documents=result.retrieved_documents,
    )
