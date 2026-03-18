import re
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from wongnai_qa.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    DICT_PATH,
    INDEX_SAMPLE_SIZE,
    QUERY_LABELS_ALGO_PATH,
    QUERY_LABELS_JUDGES_PATH,
)


QUERY_TAG_GROUPS = {
    "cuisine": {
        "thai": ["ไทย", "thai", "อาหารไทย", "ต้มยำ", "ส้มตำ", "กะเพรา", "อีสาน", "ลาบ", "น้ำพริก", "คอหมูย่าง"],
        "chinese": ["จีน", "chinese", "ติ่มซำ", "บะหมี่", "เป็ดปักกิ่ง", "แคระ", "อาหารจีน"],
        "japanese": ["ญี่ปุ่น", "japanese", "ซูชิ", "ราเมง", "ซาชิมิ", "อิซากายะ", "sushi", "salmon", "แซลมอน"],
        "indian": ["อินเดีย", "indian", "โรตี", "แกงกะหรี่", "biryani", "naan"],
        "italian": ["อิตาลี", "italian", "pizza", "พิซซ่า", "พาสต้า", "risotto", "สปาเก็ตตี้"],
        "fusion": ["ฟิวชัน", "fusion", "ร่วมสมัย", "creative"],
    },
    "food_type": {
        "seafood": ["อาหารทะเล", "ซีฟู้ด", "seafood", "ปู", "กุ้ง", "ปลา", "หอย", "ปลาหมึก", "กุ้งเผา"],
        "meat": ["หมู", "ไก่", "เนื้อ", "เนื้อวัว", "หมูกรอบ", "เป็ด", "ไก่ทอด"],
        "pizza": ["พิซซ่า", "pizza"],
        "bakery": ["เบเกอรี่", "bakery", "ขนมปัง", "croissant", "เค้ก", "cake", "toast", "โทสต์"],
        "dessert_drink": ["ของหวาน", "ขนม", "เครื่องดื่ม", "dessert", "coffee", "ชา", "คาเฟ่", "กาแฟ", "น้ำเปล่า", "ชาเขียว", "ช็อคโกแลต", "chocolate"],
        "ice_cream": ["ไอศกรีม", "ice cream", "gelato"],
        "curry_rice": ["ข้าวแกง", "แกงราดข้าว", "ข้าวราดแกง"],
        "rice_porridge": ["ข้าวต้ม", "porridge", "congee"],
        "noodle": ["ก๋วยเตี๋ยว", "บะหมี่", "ราเมง", "noodle", "ขนมจีน", "เส้น"],
        "made_to_order": ["ตามสั่ง", "อาหารตามสั่ง", "ผัดกะเพรา"],
        "healthy": ["สุขภาพ", "healthy", "คลีน", "สลัด", "อกไก่", "salad"],
        "snack": ["ของทานเล่น", "ลูกชิ้น", "ปอเปี๊ยะ", "ทอดมัน", "ยำ"],
    },
    "ambience": {
        "luxury": ["หรู", "หรูหรา", "luxury", "fine dining", "พรีเมียม"],
        "air_conditioned": ["ติดแอร์", "แอร์", "air", "air-conditioned", "ในห้าง", "ห้องแอร์"],
        "open_air": ["open air", "กลางแจ้ง", "open", "โอเพ่นแอร์", "รับลม"],
        "street_food": ["ร้านข้างทาง", "street food", "รถเข็น", "เพิง", "ริมทาง", "ริมถนน"],
        "quiet": ["สงบ", "เงียบ", "ชิล", "ชิลๆ", "ผ่อนคลาย", "นั่งสบาย", "สบายๆ"],
        "romantic": ["เดต", "โรแมนติก", "romantic", "วิวสวย"],
    },
    "price": {
        "expensive": ["แพง", "ราคาแรง", "แพงมาก", "พรีเมียม", "ราคาสูง"],
        "budget": ["ย่อมเยา", "ประหยัด", "ถูก", "ไม่แพง", "คุ้ม", "คุ้มค่า", "ราคาถูก"],
    },
    "location": {
        "beach": ["ติดทะเล", "ชายหาด", "ริมหาด", "beach", "sea view"],
        "mountain": ["บนเขา", "ภูเขา", "ดอย", "mountain"],
        "chiang_mai": ["เชียงใหม่", "chiang mai", "นิมมาน", "สันกำแพง"],
        "pattaya": ["พัทยา", "pattaya", "จอมเทียน"],
        "bangkok": ["กรุงเทพ", "bangkok", "กทม", "สยาม", "สุขุมวิท", "ทองหล่อ"],
        "riverside": ["ริมแม่น้ำ", "ริมน้ำ", "river"],
        "mall": ["ห้าง", "central", "เซ็นทรัล", "paragon", "พารากอน", "bts"],
    },
}

QUERY_STOPWORDS = {
    "ร้าน", "ร้านอาหาร", "อาหาร", "มี", "ไหม", "หน่อย", "ที่", "และ", "หรือ", "แบบ", "แถว", "อำเภอ", "เมือง", "จังหวัด",
    "ค่ะ", "ครับ", "นะคะ", "นะ", "เลย", "มาก", "อร่อย", "ดี", "รสชาติ", "ราคา", "เมนู", "จาน", "วัน", "ไป", "มา", "ของ",
    "กิน", "ทาน", "ตัว", "คน", "ๆ", "฿", "บาท", "อย่าง", "คือ", "ชั้น", "สรุป", "แต่", "ให้", "โต๊ะ", "นาที", "บรรยากาศ",
    "ดาว", "พนักงาน", "บริการ", "พนง", "สาขา", "พิกัด", "ชุด", "โดยรวม", "แนะนำ", "แล้ว", "กับ", "น้ำ", "ได้", "เป็น"
}

@dataclass(frozen=True)
class ResourceBundle:
    food_terms: set[str]
    algo_terms: set[str]
    judge_terms: set[str]
    frequent_query_terms: set[str]


def normalize_text(text: str) -> str:
    text = str(text).replace("\ufeff", " ").replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_text_lines(path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [normalize_text(line) for line in handle if normalize_text(line)]


def _parse_labeled_query_terms(path, min_term_length: int = 2) -> list[str]:
    terms: list[str] = []
    for raw_line in _read_text_lines(path):
        parts = [normalize_text(part) for part in raw_line.split("|")]
        for part in parts:
            candidate = part.lower()
            if len(candidate) < min_term_length or candidate.isdigit():
                continue
            if not _is_valid_term(candidate):
                continue
            terms.append(candidate)
    return terms


@lru_cache(maxsize=1)
def load_resource_bundle() -> ResourceBundle:
    food_terms = {
        line.lower()
        for line in _read_text_lines(DICT_PATH)
        if _is_valid_term(line.lower())
    }
    algo_terms = set(_parse_labeled_query_terms(QUERY_LABELS_ALGO_PATH))
    judge_terms = set(_parse_labeled_query_terms(QUERY_LABELS_JUDGES_PATH))
    merged = Counter(list(algo_terms) + list(judge_terms))
    frequent_query_terms = {
        term
        for term, count in merged.items()
        if count >= 3 and _is_valid_term(term)
    }
    return ResourceBundle(
        food_terms=food_terms,
        algo_terms=algo_terms,
        judge_terms=judge_terms,
        frequent_query_terms=frequent_query_terms,
    )


def load_custom_dictionary(limit: int | None = None) -> list[str]:
    words = list(load_resource_bundle().food_terms)
    return words[:limit] if limit else words


def _extract_group_hits(text: str, groups: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for label, keywords in groups.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            hits.append(label)
    return hits


def _extract_known_terms(text: str, resource_bundle: ResourceBundle, limit: int = 8) -> list[str]:
    lowered = text.lower()
    candidate_terms = resource_bundle.food_terms.union(resource_bundle.frequent_query_terms)
    hits = [
        term
        for term in candidate_terms
        if 4 <= len(term) <= 40 and term in lowered and _is_valid_term(term)
    ]
    hits.sort(key=len, reverse=True)
    filtered: list[str] = []
    for term in hits:
        if any(term in existing for existing in filtered):
            continue
        filtered.append(term)
        if len(filtered) >= limit:
            break
    return filtered


def _extract_query_tokens(text: str, limit: int = 10) -> list[str]:
    tokens = [token.lower() for token in re.findall(r"[\w\u0E00-\u0E7F]+", text)]
    filtered: list[str] = []
    for token in tokens:
        if len(token) < 2 or token.isdigit() or token in QUERY_STOPWORDS:
            continue
        if not _is_valid_term(token):
            continue
        if token in filtered:
            continue
        filtered.append(token)
        if len(filtered) >= limit:
            break
    return filtered


def _is_valid_term(term: str) -> bool:
    normalized = normalize_text(term)
    if len(normalized) < 2:
        return False
    if normalized in QUERY_STOPWORDS:
        return False
    if not re.search(r"[A-Za-z\u0E00-\u0E7F]", normalized):
        return False
    if re.search(r"^[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]", normalized):
        return False
    if re.search(r"[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]$", normalized):
        return False
    if re.fullmatch(r"[_\W]+", normalized):
        return False
    return True


def build_review_metadata(
    review_text: str,
    rating: Any,
    review_id: int,
    resource_bundle: ResourceBundle,
) -> dict[str, Any]:
    normalized = normalize_text(review_text)
    metadata: dict[str, Any] = {
        "review_id": int(review_id),
        "rating": float(rating),
        "review_length": len(normalized),
    }
    for group_name, groups in QUERY_TAG_GROUPS.items():
        metadata[group_name] = "|".join(_extract_group_hits(normalized, groups))
    metadata["known_terms"] = "|".join(_extract_known_terms(normalized, resource_bundle))
    return metadata


def load_review_dataframe(sample_size: int | None = INDEX_SAMPLE_SIZE) -> pd.DataFrame:
    dataframe = pd.read_csv(
        DATA_PATH,
        sep=";",
        on_bad_lines="skip",
        header=None,
        names=["review", "rating"],
    )
    dataframe = dataframe.dropna().copy()
    dataframe["review"] = dataframe["review"].map(normalize_text)
    dataframe = dataframe[dataframe["review"] != ""]
    dataframe["rating"] = pd.to_numeric(dataframe["rating"], errors="coerce")
    dataframe = dataframe.dropna(subset=["rating"]).reset_index(drop=True)
    if sample_size:
        dataframe = dataframe.head(sample_size).copy()
    return dataframe


def load_and_preprocess_data(sample_size: int | None = INDEX_SAMPLE_SIZE) -> list[Document]:
    print("Loading and preprocessing Wongnai reviews...")
    started_at = time.perf_counter()
    resource_bundle = load_resource_bundle()
    dataframe = load_review_dataframe(sample_size=sample_size)
    print(f"Loaded {len(dataframe)} reviews for preprocessing.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    documents: list[Document] = []
    progress_interval = max(1, len(dataframe) // 10) if len(dataframe) else 1
    for review_id, row in dataframe.iterrows():
        metadata = build_review_metadata(
            review_text=row["review"],
            rating=row["rating"],
            review_id=review_id,
            resource_bundle=resource_bundle,
        )
        chunks = splitter.split_text(row["review"])
        for chunk_id, chunk in enumerate(chunks):
            chunk_text = normalize_text(chunk)
            if not chunk_text:
                continue
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_id"] = chunk_id
            documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        if (review_id + 1) % progress_interval == 0 or (review_id + 1) == len(dataframe):
            elapsed = time.perf_counter() - started_at
            print(
                f"Preprocessed {review_id + 1}/{len(dataframe)} reviews "
                f"into {len(documents)} chunks in {elapsed:.1f}s"
            )

    total_elapsed = time.perf_counter() - started_at
    print(
        f"Prepared {len(documents)} review chunks from {len(dataframe)} reviews "
        f"in {total_elapsed:.1f}s."
    )
    return documents


def analyze_query(query: str, resource_bundle: ResourceBundle | None = None) -> dict[str, Any]:
    resource_bundle = resource_bundle or load_resource_bundle()
    normalized = normalize_text(query)
    detected_tags = {
        group_name: _extract_group_hits(normalized, groups)
        for group_name, groups in QUERY_TAG_GROUPS.items()
    }
    query_terms = _extract_known_terms(normalized, resource_bundle)
    query_tokens = _extract_query_tokens(normalized)
    expansion_terms = []
    for values in detected_tags.values():
        expansion_terms.extend(values)
    expansion_terms.extend(query_terms)
    expansion_terms.extend(query_tokens)
    expanded_query = " ".join(
        part for part in [normalized, " ".join(expansion_terms)] if part.strip()
    )
    return {
        "raw_query": query,
        "normalized_query": normalized,
        "expanded_query": expanded_query,
        "detected_tags": detected_tags,
        "query_terms": query_terms,
        "query_tokens": query_tokens,
    }
