from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from wongnai_qa.config import (
    HF_LOCAL_FILES_ONLY,
    INDEX_SAMPLE_SIZE,
    LLM_ADAPTER_DIR,
    LLM_LEARNING_RATE,
    LLM_MAX_SEQ_LENGTH,
    LLM_MODEL_NAME,
    LLM_SFT_DATA_PATH,
    LLM_TRAIN_BATCH_SIZE,
    LLM_TRAIN_EPOCHS,
    LLM_GRAD_ACCUM,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    RETRIEVER_K,
    resolve_cached_model_path,
)
from wongnai_qa.evaluation import build_benchmark
from wongnai_qa.preprocessing import analyze_query, load_and_preprocess_data, load_resource_bundle
from wongnai_qa.retrieval import rank_documents_by_profile


PROMPT_TEMPLATE = """คุณเป็นผู้ช่วยแนะนำร้านอาหารจากรีวิว Wongnai
จงตอบเป็นภาษาไทย โดยใช้ข้อมูลจากรีวิวที่ให้มาเท่านั้น
ถ้ามีหลายตัวเลือกให้ตอบเป็นข้อ และระบุ rating ทุกข้อ
ถ้าข้อมูลยังไม่ชัด ให้บอกว่าหลักฐานมีจำกัด

คำถาม:
{question}

ข้อมูลรีวิว:
{context}

คำตอบ:
"""


def _format_docs_for_training(documents: list[Any]) -> str:
    sections = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        tags = []
        for label in ["cuisine", "food_type", "ambience", "price", "location"]:
            raw_value = str(metadata.get(label, "")).strip("|")
            if raw_value:
                tags.append(f"{label}={raw_value.replace('|', ', ')}")
        sections.append(
            "\n".join(
                [
                    f"[รีวิว {index}]",
                    f"rating: {metadata.get('rating', 'N/A')}",
                    f"tags: {', '.join(tags) if tags else 'none'}",
                    f"text: {document.page_content}",
                ]
            )
        )
    return "\n\n".join(sections)


def _build_target_answer(question: str, documents: list[Any]) -> str:
    if not documents:
        return "ไม่พบข้อมูลรีวิวที่ตรงกับคำถามนี้ในชุดข้อมูล"

    lines = [f"คำแนะนำสำหรับคำถาม: {question}"]
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        tags = []
        for label in ["cuisine", "food_type", "ambience", "price", "location"]:
            raw_value = str(metadata.get(label, "")).strip("|")
            if raw_value:
                tags.append(f"{label}: {raw_value.replace('|', ', ')}")

        excerpt = document.page_content[:180].strip()
        if len(document.page_content) > 180:
            excerpt += "..."
        lines.append(
            f"- ตัวเลือกที่ {index}: rating {metadata.get('rating', 'N/A')} ดาว | "
            f"{' | '.join(tags) if tags else 'ไม่มี tag ชัดเจน'} | "
            f"หลักฐาน: {excerpt}"
        )
    lines.append("สรุป: เลือกจากตัวเลือกที่ rating และ tag ตรงกับคำถามมากที่สุด")
    return "\n".join(lines)


def build_sft_examples(
    sample_size: int = INDEX_SAMPLE_SIZE,
    benchmark_limit: int = 64,
    top_k: int = RETRIEVER_K,
) -> list[dict[str, Any]]:
    resource_bundle = load_resource_bundle()
    documents = load_and_preprocess_data(sample_size=sample_size)
    benchmark_examples = build_benchmark(limit=benchmark_limit)

    examples: list[dict[str, Any]] = []
    for example in benchmark_examples:
        query_profile = analyze_query(example.query, resource_bundle=resource_bundle)
        top_documents = rank_documents_by_profile(
            documents,
            query_profile=query_profile,
            k=top_k,
        )
        context = _format_docs_for_training(top_documents)
        answer = _build_target_answer(example.query, top_documents)
        prompt = PROMPT_TEMPLATE.format(question=example.query, context=context)
        examples.append(
            {
                "query": example.query,
                "source": example.source,
                "prompt": prompt,
                "response": answer,
                "completion": answer,
                "text": prompt + answer,
            }
        )
    return examples


def save_sft_dataset(
    output_path: str | Path = LLM_SFT_DATA_PATH,
    sample_size: int = INDEX_SAMPLE_SIZE,
    benchmark_limit: int = 64,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = build_sft_examples(sample_size=sample_size, benchmark_limit=benchmark_limit)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return output_path


def load_sft_dataset(path: str | Path = LLM_SFT_DATA_PATH) -> Dataset:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def _load_quantized_model():
    if not torch.cuda.is_available():
        raise RuntimeError("LLM fine-tuning requires CUDA.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model_path = resolve_cached_model_path(LLM_MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    return model, tokenizer


def train_lora_adapter(
    dataset_path: str | Path = LLM_SFT_DATA_PATH,
    output_dir: str | Path = LLM_ADAPTER_DIR,
) -> Path:
    model, tokenizer = _load_quantized_model()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    dataset = load_sft_dataset(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=LLM_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=LLM_GRAD_ACCUM,
        learning_rate=LLM_LEARNING_RATE,
        num_train_epochs=LLM_TRAIN_EPOCHS,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        max_length=LLM_MAX_SEQ_LENGTH,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def adapter_exists(path: str | Path = LLM_ADAPTER_DIR) -> bool:
    path = Path(path)
    return path.exists() and (path / "adapter_config.json").exists()


def load_model_with_adapter(base_model, adapter_dir: str | Path = LLM_ADAPTER_DIR):
    adapter_dir = Path(adapter_dir)
    if not adapter_exists(adapter_dir):
        return base_model
    return PeftModel.from_pretrained(base_model, str(adapter_dir))
