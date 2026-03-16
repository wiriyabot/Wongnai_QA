# Wongnai Restaurant QA: เอกสารสรุปการทำงานของระบบแบบละเอียด

## 1. วัตถุประสงค์

โปรเจกต์นี้เป็นระบบตอบคำถามเกี่ยวกับร้านอาหารจากข้อมูลรีวิว Wongnai โดยรองรับการใช้งาน 3 ช่องทางหลัก:

- `CLI` ผ่าน `main.py`
- `FastAPI` backend ผ่าน `wongnai_qa/api.py`
- `Streamlit` frontend ผ่าน `wongnai_qa/ui.py`

ระบบจะรับคำถามภาษาธรรมชาติจากผู้ใช้ จากนั้นวิเคราะห์เจตนาและคำสำคัญ ค้นหา review chunk ที่เกี่ยวข้องจาก Chroma vector store จัดอันดับผลลัพธ์ใหม่ด้วย scoring ที่อาศัย metadata และสามารถสร้างคำตอบภาษาไทยด้วย LLM ได้

## 2. ภาพรวมสถาปัตยกรรม

ระบบแบ่งเป็นชั้นหลักดังนี้:

1. `wongnai_qa/config.py`
   รวมค่าตั้งต้นของระบบ เช่น path, model, retrieval parameters, chunk settings และ environment defaults

2. `wongnai_qa/preprocessing.py`
   โหลดข้อมูลรีวิวและ resource ต่าง ๆ, ทำ normalization, สร้าง metadata tags, split รีวิวเป็น chunk และวิเคราะห์ query ของผู้ใช้

3. `wongnai_qa/retrieval.py`
   สร้างหรือโหลด Chroma vector store, สร้าง embeddings ด้วย multilingual E5, ค้นหาเอกสารด้วย similarity search และ rerank ด้วย weighted scoring

4. `wongnai_qa/generation.py`
   จัดรูปแบบ evidence, สร้าง prompt สำหรับตอบคำถาม, สร้าง baseline answer และสร้าง improved answer ด้วย LLM

5. `wongnai_qa/llm.py`
   โหลดโมเดลภาษาสำหรับ generation และตั้งค่า inference

6. `wongnai_qa/service.py`
   เป็น orchestration layer กลางที่ใช้ร่วมกันระหว่าง CLI, API และ UI

7. `wongnai_qa/api.py`
   เปิด endpoint ของ FastAPI สำหรับ health check, demo queries และ query answering

8. `wongnai_qa/ui.py`
   เป็นส่วนติดต่อผู้ใช้สำหรับกรอกคำถามและดูคำตอบพร้อมหลักฐานอ้างอิง

## 3. แหล่งข้อมูลและ resource ที่ใช้

ระบบใช้ไฟล์ภายใต้ `review_dataset/` เป็นหลัก:

- `w_review_train.csv`
  ชุดข้อมูลรีวิวหลัก แต่ละแถวเก็บข้อความรีวิวและคะแนน rating

- `food_dictionary.txt`
  พจนานุกรมคำเกี่ยวกับอาหารที่ใช้เป็น lexical resource

- `labeled_queries_by_algo.txt`
  ชุดคำ query ที่มาจากการ label โดยอัลกอริทึม

- `labeled_queries_by_judges.txt`
  ชุดคำ query ที่มาจากการ label โดยมนุษย์

ไฟล์เหล่านี้จะถูกรวมเป็น `ResourceBundle` ใน `wongnai_qa/preprocessing.py`

## 4. การตั้งค่าและค่าเริ่มต้นของระบบ

ค่าหลักของระบบถูกกำหนดใน `wongnai_qa/config.py`

### Paths

- โฟลเดอร์ dataset: `review_dataset/`
- โฟลเดอร์ vector DB: `chroma_db/`
- ไฟล์ metadata ของ index: `chroma_db/index_meta.json`

### Models

- Embedding model: `intfloat/multilingual-e5-large`
- LLM model: `scb10x/typhoon-7b`

### Retrieval Defaults

- Sample size: `3000`
- Chunk size: `700`
- Chunk overlap: `120`
- จำนวนผลลัพธ์สุดท้าย (`top-k`): `4`
- จำนวน candidate ที่ดึงมาก่อน rerank (`fetch-k`): `12`

### Scoring Weights

ใช้ในขั้น reranking:

- Rating weight: `0.12`
- Tag match weight: `0.45`
- Keyword match weight: `0.33`
- Exact phrase weight: `0.10`

### การทำงานแบบ offline

ระบบตั้งค่า environment flags เหล่านี้เป็นค่าเริ่มต้น:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

ดังนั้นในสภาพแวดล้อมปัจจุบันจะคาดหวังว่า model ต้องถูกดาวน์โหลดไว้ล่วงหน้าแล้ว เว้นแต่จะมีการเปลี่ยน environment variable

## 5. ลำดับการทำงานของระบบตั้งแต่ต้นจนจบ

ส่วนนี้อธิบาย flow ตั้งแต่ผู้ใช้ส่งคำถามเข้ามาจนได้คำตอบกลับออกไป

### ขั้นที่ 1: ผู้ใช้ส่งคำถาม

คำถามสามารถเข้าระบบได้จาก:

- argument ของ CLI ใน `main.py`
- POST `/query` ใน `wongnai_qa/api.py`
- หน้า Streamlit ใน `wongnai_qa/ui.py`

ท้ายที่สุดทุกช่องทางจะเข้าสู่:

- `get_service(...)`
- `WongnaiQAService.query(...)`

### ขั้นที่ 2: การสร้าง service

`wongnai_qa/service.py` ใช้ shared singleton service:

- `_service` เก็บ instance ปัจจุบันของ `WongnaiQAService`
- `_service_lock` ใช้ป้องกัน race condition ขณะสร้าง service

ภายใน `WongnaiQAService` จะมี:

- `_resource_bundle` โหลดครั้งเดียว
- `_vector_store` cache ไว้หลังจากสร้างหรือโหลดครั้งแรก
- `_llm` โหลดแบบ lazy เฉพาะตอนต้องใช้
- `_lock` ใช้ให้การสร้าง vector store และ model เป็น thread-safe

### ขั้นที่ 3: ตรวจสอบความพร้อมของ vector store

`ensure_vector_store()` จะตัดสินใจว่าจะ:

- ใช้ Chroma index เดิม
- หรือ rebuild index ใหม่จาก documents

ระบบจะ rebuild ถ้า:

- `rebuild=True`
- `_vector_store` ยังไม่ถูกสร้าง
- metadata ของ index บนดิสก์ไม่ตรงกับ config ปัจจุบัน

การตรวจสอบ metadata จะใช้ข้อมูลต่อไปนี้:

- index version
- embedding model name
- sample size
- collection name

จุดประสงค์คือป้องกันการเอา index เก่าที่สร้างจาก config คนละชุดมาใช้ต่อ

### ขั้นที่ 4: โหลดข้อมูลดิบและทำ normalization

เมื่อจำเป็นต้อง rebuild ระบบจะเรียก `load_and_preprocess_data()`

สิ่งที่ทำมีดังนี้:

1. อ่าน `w_review_train.csv`
2. ลบแถวที่ข้อมูลไม่สมบูรณ์หรือ rating ใช้ไม่ได้
3. ทำ normalization ข้อความรีวิว
4. จำกัดจำนวนแถวตาม `sample_size`
5. split รีวิวแต่ละอันเป็นหลาย chunk
6. แนบ metadata ให้แต่ละ chunk

การ normalize ข้อความประกอบด้วย:

- ลบ BOM และ zero-width characters
- รวม whitespace ที่ซ้ำกัน
- ตัดช่องว่างหัวท้าย

### ขั้นที่ 5: สร้าง metadata จากรีวิว

แต่ละ review chunk จะมี metadata ที่สร้างโดย `build_review_metadata()`

metadata หลักประกอบด้วย:

- `review_id`
- `rating`
- `review_length`
- `cuisine`
- `food_type`
- `ambience`
- `price`
- `location`
- `known_terms`

tag เหล่านี้อิงจาก keyword dictionaries ใน `QUERY_TAG_GROUPS`

ตัวอย่าง:

- ถ้ารีวิวมีคำว่า `ซีฟู้ด`, `กุ้ง`, `ปลา`
  ระบบอาจใส่ `food_type=seafood`

- ถ้ามีคำว่า `พัทยา`
  ระบบอาจใส่ `location=pattaya`

metadata เหล่านี้จะเก็บเป็น string ที่คั่นด้วย `|`

### ขั้นที่ 6: การ split รีวิวเป็น chunk

ระบบใช้ `RecursiveCharacterTextSplitter` โดยตั้งค่า:

- `chunk_size=700`
- `chunk_overlap=120`

แต่ละ chunk จะกลายเป็น `Document` object ที่มี:

- `page_content`
- `metadata`

วิธีนี้ช่วยให้ retrieval ละเอียดกว่าใช้รีวิวเต็มทั้งก้อนเพียงอย่างเดียว

### ขั้นที่ 7: การสร้าง embeddings และ vector index

`wongnai_qa/retrieval.py` นิยาม `E5Embeddings` ซึ่งครอบ `HuggingFaceEmbeddings` อีกชั้น

รายละเอียดสำคัญ:

- ฝั่ง document จะ embed ด้วย prefix `passage:`
- ฝั่ง query จะ embed ด้วย prefix `query:`

วิธีนี้สอดคล้องกับรูปแบบการใช้งานของโมเดล E5

vector index จะถูกสร้างใน Chroma โดยอิง:

- collection name จาก config
- persist directory ที่ `chroma_db/`

หลังสร้างเสร็จ ระบบจะเขียน metadata ของ index ไปที่:

- `chroma_db/index_meta.json`

## 6. การวิเคราะห์ query ของผู้ใช้

ก่อน retrieval ระบบจะเรียก `analyze_query()`

ผลลัพธ์คือ `query_profile` ซึ่งมีข้อมูล:

- `raw_query`
- `normalized_query`
- `expanded_query`
- `detected_tags`
- `query_terms`

### Detected tags

ระบบจะเช็ก query เทียบกับกลุ่ม tag ต่อไปนี้:

- `cuisine`
- `food_type`
- `ambience`
- `price`
- `location`

ตัวอย่าง:

- "ร้านญี่ปุ่นคุ้มราคาแถวเชียงใหม่"
  อาจตรวจพบ:
  - `cuisine=japanese`
  - `price=budget`
  - `location=chiang_mai`

### Query terms

ระบบยังตรวจด้วยว่ามีคำจาก resource dictionaries ปรากฏใน query หรือไม่

คำพวกนี้จะถูก:

- แปลงเป็น lowercase
- กรองตามความยาวขั้นต่ำ
- เรียงตามความยาว
- ตัดคำซ้ำหรือคำที่ซ้อนกันมากเกินไป

### Expanded query

ข้อความสุดท้ายที่ใช้ค้นใน vector store คือ:

- normalized query
- รวมกับชื่อ tag ที่ตรวจพบ
- รวมกับ known query terms ที่ match

จุดประสงค์คือเพิ่ม recall ของ semantic retrieval

## 7. Retrieval และ Reranking

ขั้น retrieval อยู่ใน `retrieve_documents()`

### ระยะที่ 1: Similarity search

ระบบจะเรียก Chroma เพื่อดึง candidate `fetch_k` รายการโดยใช้:

- `query_profile["expanded_query"]`

ค่าปกติคือ:

- ดึง candidate 12 รายการ
- คัดเหลือ 4 รายการสุดท้าย

### ระยะที่ 2: Weighted reranking

candidate แต่ละตัวจะถูกคำนวณคะแนนใหม่ด้วย `_score_document()`

คะแนนรวมมาจาก 4 สัญญาณหลัก:

1. Review rating
   รีวิวที่ได้ดาวสูงกว่าจะมีคะแนนพื้นฐานมากกว่า

2. Metadata tag overlap
   ถ้า query ขอคุณสมบัติอย่าง `japanese`, `budget`, `pattaya`
   เอกสารที่ metadata ตรงจะได้คะแนนเพิ่มมาก

3. Query term overlap
   ถ้าเอกสารมีคำสำคัญที่ตรงกับ query terms ก็จะได้คะแนนเพิ่ม

4. Exact phrase match
   ถ้าข้อความ query แบบ normalize ปรากฏตรง ๆ ใน chunk จะได้โบนัสเพิ่ม

วิธีนี้ทำให้ retrieval มีความเป็น domain-aware มากกว่า semantic similarity เพียงอย่างเดียว

### ระยะที่ 3: Deduplication

หลังจาก sort ตามคะแนนแล้ว ระบบจะ:

- ลบ document ที่ซ้ำกันตาม `(review_id, chunk_id)`
- คืนผลลัพธ์ top `k` ที่ไม่ซ้ำกัน

## 8. การสร้างคำตอบ

ระบบมีคำตอบ 2 แบบ

### 8.1 Baseline answer

`build_baseline_answer()` จะสร้างสรุปแบบ retrieval-only

ภายในประกอบด้วย:

- query เดิม
- รายการรีวิวที่ดึงมาได้
- rating
- metadata tags
- excerpt ของข้อความ

เหมาะสำหรับใช้ debug และเทียบคุณภาพ retrieval แบบตรงไปตรงมา

### 8.2 Improved RAG answer

`build_rag_answer()` จะสร้างคำตอบภาษาไทยโดยใช้:

- คำถามของผู้ใช้
- evidence ที่ถูกจัดรูปแบบแล้ว
- โมเดล LLM

prompt จะสั่งโมเดลให้:

- ตอบเป็นภาษาไทย
- ใช้เฉพาะ evidence ที่ให้มา
- ใส่ star rating ในคำตอบ
- ห้ามแต่งชื่อร้าน ที่อยู่ หรือราคา
- ถ้าไม่มีชื่อร้านในรีวิว ให้เรียกเป็น "ตัวเลือกที่ 1", "ตัวเลือกที่ 2" ฯลฯ
- กล่าวถึง cuisine / ambience / location เฉพาะเมื่อมีหลักฐานรองรับ
- ถ้าหลักฐานอ่อนหรือไม่ครบ ให้บอกอย่างสั้น ๆ
- ถ้าไม่มีข้อมูลรองรับเลย ให้บอกว่าไม่พบข้อมูลที่ตรงใน dataset

ถ้าไม่พบเอกสารที่เกี่ยวข้องเลย ทั้ง baseline และ improved answer จะคืน fallback message

## 9. การโหลด LLM และข้อจำกัด

`wongnai_qa/llm.py` จะโหลด LLM เฉพาะเมื่อ `include_improved=True`

ลำดับการทำงาน:

- โหลด tokenizer ก่อน
- ตรวจว่ามี CUDA หรือไม่
- โหลด model แบบ 4-bit ด้วย `BitsAndBytesConfig`
- ครอบด้วย Hugging Face `pipeline("text-generation")`
- แปลงเป็น `HuggingFacePipeline`

generation settings หลัก:

- max new tokens: `384`
- temperature: `0.1`
- repetition penalty: `1.1`
- ปิด sampling

ข้อจำกัดปัจจุบัน:

- ถ้าไม่มี CUDA จะไม่สามารถใช้ improved generation ได้
- แต่ retrieval-only mode ยังใช้ได้ถ้าข้าม LLM

## 10. พฤติกรรมของ API

`wongnai_qa/api.py` เปิด 3 endpoint หลัก

### `GET /health`

คืนค่า:

```json
{"status": "ok"}
```

ใช้สำหรับ health check

### `GET /demo-queries`

คืนรายการคำถามตัวอย่างจาก `DEFAULT_QUERY_SET` ใน `wongnai_qa/config.py`

### `POST /query`

request fields:

- `query`
- `sample_size`
- `top_k`
- `fetch_k`
- `include_improved`
- `rebuild`

response fields:

- `query`
- `query_profile`
- `baseline_answer`
- `improved_answer`
- `retrieved_documents`

response ของ API ถูก serialize มาแล้วสำหรับใช้งานต่อใน UI

## 11. พฤติกรรมของ Streamlit Frontend

frontend เป็นเพียง thin client

หน้าที่หลักคือ:

- รับคำถามจากผู้ใช้
- เรียก API
- แสดงคำตอบหลัก
- แสดงรีวิวที่ใช้เป็นหลักฐานอ้างอิง

frontend ไม่ได้ทำ retrieval หรือ generation เองในเครื่อง

พฤติกรรม UI ปัจจุบัน:

- ใช้ retrieval defaults ที่กำหนดไว้ภายใน
- flow เรียบง่าย เน้น user-facing
- แสดงคำตอบเป็นศูนย์กลาง
- แสดง evidence cards จาก metadata ที่ API ส่งกลับมา

## 12. พฤติกรรมของ CLI

`main.py` เป็น terminal interface แบบง่าย

เหมาะสำหรับ:

- debug เร็ว ๆ
- ตรวจผล retrieval
- demo ระบบ
- ใช้งานโดยไม่ต้องเปิด API หรือ UI

รองรับ option:

- `--query` หลายตัว
- `--sample-size`
- `--top-k`
- `--fetch-k`
- `--rebuild`
- `--skip-llm`

## 13. Caching และประสิทธิภาพ

ระบบมี caching หลายชั้น:

- `load_resource_bundle()` ใช้ `lru_cache`
- `WongnaiQAService` cache vector store และ LLM ใน memory
- Chroma index ถูก persist ไว้บนดิสก์

สิ่งนี้ช่วยลดต้นทุนการ preprocess และการโหลด model ซ้ำ

operation ที่แพงที่สุดคือ:

- การ rebuild embeddings และ vector store
- การโหลด LLM

เส้นทางที่เร็วที่สุดคือ:

- ใช้ index เดิม
- ข้าม LLM generation

## 14. จุดเด่นและ tradeoff ของดีไซน์ปัจจุบัน

จุดแข็ง:

- โครงสร้างระบบตรงไปตรงมา
- ใช้ service กลางร่วมกันทุก interface
- reranking ใช้ metadata จริงจากโดเมน
- retrieval path ค่อนข้าง deterministic
- prompt มีข้อกำหนดชัดเจนเพื่อลด hallucination

tradeoff:

- ชื่อร้านจะมีได้ก็ต่อเมื่อรีวิวมีระบุไว้จริง
- generation ปัจจุบันพึ่ง GPU
- การวิเคราะห์ query ยังอิง keyword lists ไม่ใช่ learned classifier
- chunk-level retrieval อาจพลาด context ที่กระจายอยู่หลาย chunk
- การเปลี่ยน `sample_size` มีผลต่อผลลัพธ์ retrieval อย่างชัดเจน

## 15. สรุป execution flow แบบย่อ

สรุปหนึ่งบรรทัด:

`user query -> query analysis -> vector retrieval -> metadata reranking -> baseline summary -> optional LLM answer -> response`

ลำดับแบบขยาย:

1. โหลดหรือ reuse lexical resources
2. โหลดหรือ reuse vector store
3. วิเคราะห์ query เป็น tags และ known terms
4. ดึง candidate chunks ด้วย vector similarity
5. rerank ด้วย rating, tags และ keyword overlap
6. สร้าง baseline answer
7. ถ้าต้องการ improved answer ให้โหลด LLM และ generate คำตอบภาษาไทย
8. ส่ง response กลับไปยัง CLI, API หรือ Streamlit UI

## 16. แนวทางปรับปรุงในอนาคต

สิ่งที่ควรพัฒนาต่อ:

- เพิ่มการ extract ชื่อร้านอย่างชัดเจนตั้งแต่ขั้น preprocessing
- แยก retrieval ระดับรีวิวออกจากการสรุประดับร้าน
- รองรับ generator ที่ใช้ CPU ได้จริง
- เพิ่ม automated tests สำหรับ query analysis และ reranking
- ย้าย prompt และ scoring weights ไปเป็น config ที่ทดลองได้ง่ายขึ้น
- เก็บ retrieval scores ไว้เพื่อให้ debug ได้สะดวกขึ้น
