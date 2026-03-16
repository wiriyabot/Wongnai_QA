import re
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
        "thai": ["ไทย", "thai", "อาหารไทย", "ต้มยำ", "ส้มตำ", "กะเพรา"],
        "chinese": ["จีน", "chinese", "ติ่มซำ", "บะหมี่", "เป็ดปักกิ่ง"],
        "japanese": ["ญี่ปุ่น", "japanese", "ซูชิ", "ราเมง", "ซาชิมิ", "อิซากายะ"],
        "indian": ["อินเดีย", "indian", "โรตี", "แกงกะหรี่", "biryani", "naan"],
        "italian": ["อิตาลี", "italian", "pizza", "พิซซ่า", "พาสต้า", "risotto"],
        "fusion": ["ฟิวชัน", "fusion", "ร่วมสมัย", "creative"],
    },
    "food_type": {
        "seafood": ["อาหารทะเล", "ซีฟู้ด", "seafood", "ปู", "กุ้ง", "ปลา", "หอย"],
        "pizza": ["พิซซ่า", "pizza"],
        "bakery": ["เบเกอรี่", "bakery", "ขนมปัง", "croissant", "เค้ก"],
        "dessert_drink": ["ของหวาน", "ขนม", "เครื่องดื่ม", "dessert", "coffee", "ชา", "คาเฟ่"],
        "ice_cream": ["ไอศกรีม", "ice cream", "gelato"],
        "curry_rice": ["ข้าวแกง", "แกงราดข้าว"],
        "rice_porridge": ["ข้าวต้ม", "porridge", "congee"],
        "noodle": ["ก๋วยเตี๋ยว", "บะหมี่", "ราเมง", "noodle"],
        "made_to_order": ["ตามสั่ง", "อาหารตามสั่ง", "ผัดกะเพรา"],
        "healthy": ["สุขภาพ", "healthy", "คลีน", "สลัด", "อกไก่"],
    },
    "ambience": {
        "luxury": ["หรู", "หรูหรา", "luxury", "fine dining", "พรีเมียม"],
        "air_conditioned": ["ติดแอร์", "แอร์", "air", "air-conditioned"],
        "open_air": ["open air", "กลางแจ้ง", "open", "โอเพ่นแอร์"],
        "street_food": ["ร้านข้างทาง", "street food", "รถเข็น", "เพิง", "ริมทาง"],
        "quiet": ["สงบ", "เงียบ", "ชิล", "ชิลๆ", "ผ่อนคลาย"],
        "romantic": ["เดต", "โรแมนติก", "romantic", "วิวสวย"],
    },
    "price": {
        "expensive": ["แพง", "ราคาแรง", "แพงมาก", "พรีเมียม"],
        "budget": ["ย่อมเยา", "ประหยัด", "ถูก", "ไม่แพง", "คุ้ม"],
    },
    "location": {
        "beach": ["ติดทะเล", "ชายหาด", "ริมหาด", "beach", "sea view"],
        "mountain": ["บนเขา", "ภูเขา", "ดอย", "mountain"],
        "chiang_mai": ["เชียงใหม่", "chiang mai"],
        "pattaya": ["พัทยา", "pattaya"],
        "bangkok": ["กรุงเทพ", "bangkok", "กทม"],
        "riverside": ["ริมแม่น้ำ", "ริมน้ำ", "river"],
    },
}

QUERY_STOPWORDS = {
    "ร้าน",
    "ร้านอาหาร",
    "อาหาร",
    "มี",
    "ไหม",
    "หน่อย",
    "ที่",
    "และ",
    "หรือ",
    "แบบ",
    "แถว",
    "อำเภอ",
    "เมือง",
    "จังหวัด",
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
            if len(part) < min_term_length or part.isdigit():
                continue
            terms.append(part.lower())
    return terms


@lru_cache(maxsize=1)
def load_resource_bundle() -> ResourceBundle:
    food_terms = {line.lower() for line in _read_text_lines(DICT_PATH)}
    algo_terms = set(_parse_labeled_query_terms(QUERY_LABELS_ALGO_PATH))
    judge_terms = set(_parse_labeled_query_terms(QUERY_LABELS_JUDGES_PATH))
    merged = Counter(list(algo_terms) + list(judge_terms))
    frequent_query_terms = {term for term, count in merged.items() if count >= 1}
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
    hits = [
        term
        for term in resource_bundle.frequent_query_terms
        if len(term) >= 4 and term in lowered
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
        if token in filtered:
            continue
        filtered.append(token)
        if len(filtered) >= limit:
            break
    return filtered


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
    resource_bundle = load_resource_bundle()
    dataframe = load_review_dataframe(sample_size=sample_size)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    documents: list[Document] = []
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

    print(f"Prepared {len(documents)} review chunks from {len(dataframe)} reviews.")
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
