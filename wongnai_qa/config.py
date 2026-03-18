import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
DATASET_DIR = BASE_DIR / "review_dataset"
DATA_PATH = DATASET_DIR / "w_review_train.csv"
DICT_PATH = DATASET_DIR / "food_dictionary.txt"
QUERY_LABELS_ALGO_PATH = DATASET_DIR / "labeled_queries_by_algo.txt"
QUERY_LABELS_JUDGES_PATH = DATASET_DIR / "labeled_queries_by_judges.txt"
TEST_FILE_PATH = DATASET_DIR / "test_file.csv"
SAMPLE_SUBMISSION_PATH = DATASET_DIR / "sample_submission.csv"

VECTOR_DB_DIR = BASE_DIR / "chroma_db"
VECTOR_DB_META_PATH = VECTOR_DB_DIR / "index_meta.json"
VECTOR_COLLECTION_NAME = "wongnai_reviews_v2"
INDEX_VERSION = "2.0"

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "1") == "1"
LLM_ADAPTER_DIR = os.getenv(
    "LLM_ADAPTER_DIR",
    str(BASE_DIR / "artifacts" / "qwen2_5_7b_instruct_lora_adapter"),
)

# Default to full dataset by using a very large cap; can still be overridden via WONGNAI_SAMPLE_SIZE.
INDEX_SAMPLE_SIZE = int(os.getenv("WONGNAI_SAMPLE_SIZE", "999999"))
CHUNK_SIZE = int(os.getenv("WONGNAI_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("WONGNAI_CHUNK_OVERLAP", "120"))
RETRIEVER_K = int(os.getenv("WONGNAI_RETRIEVER_K", "4"))
RETRIEVER_FETCH_K = int(os.getenv("WONGNAI_RETRIEVER_FETCH_K", "12"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "384"))
LORA_RANK = int(os.getenv("WONGNAI_LORA_RANK", "16"))
LORA_ALPHA = int(os.getenv("WONGNAI_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("WONGNAI_LORA_DROPOUT", "0.05"))
LLM_TRAIN_EPOCHS = int(os.getenv("WONGNAI_LLM_TRAIN_EPOCHS", "1"))
LLM_TRAIN_BATCH_SIZE = int(os.getenv("WONGNAI_LLM_TRAIN_BATCH_SIZE", "1"))
LLM_GRAD_ACCUM = int(os.getenv("WONGNAI_LLM_GRAD_ACCUM", "8"))
LLM_LEARNING_RATE = float(os.getenv("WONGNAI_LLM_LEARNING_RATE", "2e-4"))
LLM_MAX_SEQ_LENGTH = int(os.getenv("WONGNAI_LLM_MAX_SEQ_LENGTH", "1024"))
LLM_SFT_DATA_PATH = BASE_DIR / "artifacts" / "llm_sft_dataset.jsonl"

RATING_WEIGHT = 0.12
TAG_MATCH_WEIGHT = 0.45
KEYWORD_MATCH_WEIGHT = 0.33
EXACT_PHRASE_WEIGHT = 0.10

DEFAULT_QUERY_SET = [
    "อยากกินซูชิหรือพาสต้า มีร้านไหนแนะนำ",
    "อยากกินซีฟู้ดหรือเค้ก มีร้านไหนน่าสนใจบ้าง",
    "หาร้านหรู ติดแอร์ ในห้าง ราคาไม่แพง",
    "มีร้านชิลๆ ติดทะเลแถวพัทยาไหม",
    "อยากได้พิซซ่าอิตาลี ติดแอร์ ในห้าง ราคาไม่แพง",
]

ASSIGNMENT_QUERY_SET = {
    "cuisine": "อยากกินซูชิหรือพาสต้า มีร้านไหนแนะนำ",
    "food_type": "อยากกินซีฟู้ดหรือเค้ก มีร้านไหนน่าสนใจบ้าง",
    "ambience_price": "หาร้านหรู ติดแอร์ ในห้าง ราคาไม่แพง",
    "location": "มีร้านชิลๆ ติดทะเลแถวพัทยาไหม",
    "mixed": "อยากได้พิซซ่าอิตาลี ติดแอร์ ในห้าง ราคาไม่แพง",
}


def resolve_cached_model_path(model_name: str) -> str:
    if Path(model_name).exists():
        return model_name

    if "/" not in model_name:
        return model_name

    org, repo = model_name.split("/", 1)
    repo_dir = HF_CACHE_DIR / f"models--{org}--{repo}"
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        snapshot_id = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = repo_dir / "snapshots" / snapshot_id
        if snapshot_dir.exists():
            return str(snapshot_dir)
    return model_name
