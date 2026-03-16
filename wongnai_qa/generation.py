from typing import Any

from langchain_core.prompts import PromptTemplate


def _format_tag_summary(metadata: dict[str, Any]) -> str:
    parts = []
    for label in ["cuisine", "food_type", "ambience", "price", "location"]:
        values = [value for value in str(metadata.get(label, "")).split("|") if value]
        if values:
            parts.append(f"{label}={', '.join(values)}")
    return " | ".join(parts) if parts else "no explicit tags"


def format_docs(documents: list) -> str:
    sections = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        sections.append(
            "\n".join(
                [
                    f"[Review {index}]",
                    f"Rating: {metadata.get('rating', 'N/A')} stars",
                    f"Tags: {_format_tag_summary(metadata)}",
                    f"Review Text: {document.page_content}",
                ]
            )
        )
    return "\n\n".join(sections)


def create_answer_prompt() -> PromptTemplate:
    template = """You are a Wongnai restaurant recommendation assistant.
Answer in Thai. Use only the supplied review evidence.

Requirements:
- Answer the user's request directly and concretely.
- If recommending multiple options, separate them into clear bullet points.
- Always include the star rating for each recommendation.
- Do not invent restaurant names, addresses, or prices.
- If the review evidence does not contain a restaurant name, label it as "ตัวเลือกที่ 1", "ตัวเลือกที่ 2", and so on.
- Mention cuisine / food type / ambience / location when the evidence supports it.
- If the evidence is weak or partial, say so briefly.
- If no supporting evidence exists, say that no matching result was found in the dataset.

User question:
{question}

Retrieved evidence:
{context}

Answer:"""
    return PromptTemplate.from_template(template)


def build_baseline_answer(question: str, documents: list) -> str:
    if not documents:
        return "Baseline: ไม่พบข้อมูลที่ตรงกับคำถามในชุดรีวิว"

    lines = ["Baseline answer", f"Query: {question}", "Top retrieved reviews:"]
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        excerpt = document.page_content[:220].strip()
        if len(document.page_content) > 220:
            excerpt += "..."
        lines.append(
            f"{index}. rating={metadata.get('rating', 'N/A')} stars | "
            f"tags={_format_tag_summary(metadata)} | excerpt={excerpt}"
        )
    return "\n".join(lines)


def build_rag_answer(question: str, documents: list, llm) -> str:
    if not documents:
        return "Improved answer: ไม่พบข้อมูลที่ตรงกับคำถามในชุดรีวิว"

    prompt = create_answer_prompt().format(
        question=question,
        context=format_docs(documents),
    )
    return llm.invoke(prompt).strip()

