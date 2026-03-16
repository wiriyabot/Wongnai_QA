from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)

from wongnai_qa.config import (
    HF_LOCAL_FILES_ONLY,
    LLM_ADAPTER_DIR,
    LLM_MODEL_NAME,
    MAX_NEW_TOKENS,
    resolve_cached_model_path,
)


@dataclass
class LocalGenerator:
    text_pipeline: any
    tokenizer: any
    system_prompt: str = "You are a helpful multilingual assistant. Follow the user's instructions carefully and answer in Thai when requested."

    def invoke(self, prompt: str) -> str:
        formatted_prompt = prompt
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        result = self.text_pipeline(formatted_prompt)
        if isinstance(result, list) and result:
            return str(result[0].get("generated_text", "")).strip()
        return str(result).strip()


def load_llm():
    """
    Load the generation model used by the RAG pipeline.
    The default model is configured for 4-bit GPU inference.
    """
    print(f"Loading LLM: {LLM_MODEL_NAME}...")
    model_path = resolve_cached_model_path(LLM_MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{LLM_MODEL_NAME} is configured for GPU inference, but no CUDA device was found. "
            "This machine reports torch.cuda.is_available() == False. "
            "Use a GPU, or change LLM_MODEL_NAME to a smaller CPU-friendly model."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    from wongnai_qa.finetuning import adapter_exists, load_model_with_adapter

    if adapter_exists(LLM_ADAPTER_DIR):
        try:
            print(f"Loading LoRA adapter from {LLM_ADAPTER_DIR}")
            model = load_model_with_adapter(model, LLM_ADAPTER_DIR)
        except Exception as exc:
            print(f"Skipping LoRA adapter at {LLM_ADAPTER_DIR}: {exc}")

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.max_new_tokens = MAX_NEW_TOKENS
    generation_config.temperature = 0.1
    generation_config.repetition_penalty = 1.1
    generation_config.do_sample = False

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.max_length = None
    model.generation_config = generation_config

    text_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    return LocalGenerator(text_pipeline=text_pipe, tokenizer=tokenizer)
