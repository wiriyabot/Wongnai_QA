# รายงานสรุปแนวคิดการทำงานสำหรับส่งอาจารย์

## 1. เป้าหมายของงาน

โปรเจกต์นี้มีเป้าหมายเพื่อใช้ข้อมูล `wongnai-review-dataset` ในการตอบคำถามเกี่ยวกับอาหารและร้านอาหาร โดยรองรับคำถามอย่างน้อย 5 ด้าน:

1. สัญชาติอาหาร
2. ประเภทอาหาร
3. บรรยากาศร้านและราคา
4. สถานที่ตั้ง
5. คำถามแบบผสมหลายเงื่อนไข

ระบบนี้ใช้ทั้งส่วน retrieval และ generative QA ตามโจทย์ และสามารถแสดงผลเปรียบเทียบระหว่าง `baseline model` กับ `finetuned model` ได้

## 2. ข้อมูลที่ใช้

ข้อมูลหลักมาจาก:

- `review_dataset/w_review_train.csv`
- `review_dataset/food_dictionary.txt`
- `review_dataset/labeled_queries_by_algo.txt`
- `review_dataset/labeled_queries_by_judges.txt`

แนวคิดสำคัญคือไม่ได้ใช้แค่รีวิวอย่างเดียว แต่ยังใช้ query labels และ food dictionary เพื่อช่วยให้ระบบเข้าใจคำถามในเชิง domain มากขึ้น

## 3. การเตรียมข้อมูล (Data Preprocessing)

การเตรียมข้อมูลเป็นส่วนสำคัญมากสำหรับงานนี้ โดยระบบทำดังนี้:

1. อ่านรีวิวจากไฟล์ CSV
2. ลบข้อมูลว่างและคะแนนที่ไม่ถูกต้อง
3. ทำ normalization ข้อความ
4. ตัดรีวิวเป็นหลาย chunk
5. สร้าง metadata ให้แต่ละ chunk

metadata ที่สร้างประกอบด้วย:

- rating
- cuisine
- food_type
- ambience
- price
- location
- known_terms

นอกจากนี้ยังใช้ข้อมูลจาก `food_dictionary.txt` และ query labels มาสร้าง `ResourceBundle` เพื่อช่วยในการวิเคราะห์ query ของผู้ใช้

## 4. การเลือกโมเดล

### 4.1 Embedding Model

ใช้ `intfloat/multilingual-e5-large`

เหตุผล:

- รองรับหลายภาษา
- ใช้ได้กับทั้งภาษาไทยและภาษาอังกฤษ
- เหมาะกับ semantic retrieval

### 4.2 Generation Model

ใช้ `scb10x/typhoon-7b`

เหตุผล:

- เหมาะกับการสร้างคำตอบภาษาไทย
- ใช้ทำ abstractive summarization / generative QA ได้

## 5. Baseline Model

baseline model ในงานนี้คือ:

- ใช้ vector similarity retrieval จาก query ที่ normalize แล้ว
- ไม่ใช้ query expansion
- ไม่ใช้ metadata-aware reranking

ผลที่ได้จะเป็นรายการรีวิวที่ใกล้เคียงกับ query ตาม embedding space และระบบจะสรุปออกมาเป็น baseline retrieval summary

## 6. Finetuned Model

finetuned model ในโปรเจกต์นี้เป็น `domain-tuned retriever/reranker`

แนวคิดคือ:

1. ใช้ query labels จาก `labeled_queries_by_algo.txt` และ `labeled_queries_by_judges.txt`
2. สร้าง benchmark queries
3. ใช้ metadata และ known terms ของระบบเป็น pseudo-supervision
4. tune น้ำหนักของ scoring function สำหรับ retriever/reranker

น้ำหนักที่ถูกปรับประกอบด้วย:

- rating weight
- tag match weight
- keyword match weight
- exact phrase weight

ดังนั้นคำว่า `finetuned` ในโปรเจกต์นี้หมายถึง:

- การปรับ retriever/reranker ให้เข้ากับโดเมน Wongnai ด้วย labeled queries
- ไม่ใช่การ fine-tune LLM แบบ full parameter หรือ LoRA

อย่างไรก็ตาม ในรอบพัฒนาล่าสุดได้เพิ่มส่วน `LLM fine-tuning` แบบ `QLoRA/LoRA` เข้าไปเพิ่มเติมด้วย โดยแยกออกจากส่วน retriever อย่างชัดเจน

## 7. การวัดผล (Evaluation)

มีสคริปต์สำหรับประเมินผล:

- `scripts/tune_retriever.py`
- `scripts/evaluate_models.py`

metrics ที่ใช้เปรียบเทียบ baseline และ finetuned มีอย่างน้อย:

- `hit_rate_at_k`
- `avg_relevant_ratio_at_k`

นิยาม relevance ในงานนี้อิงจาก:

- metadata tags ที่ตรงกับ query
- query terms ที่ตรงกับเนื้อหาเอกสาร

แนวคิดนี้เป็น pseudo-label evaluation ซึ่งเหมาะกับงานที่ไม่มี gold relevance labels แบบสมบูรณ์

## 8. การตอบคำถามด้วย Generative QA

หลังจากได้เอกสารจาก finetuned retriever แล้ว ระบบจะส่งเอกสารเหล่านั้นเข้า LLM เพื่อสร้างคำตอบภาษาไทย

prompt กำหนดให้:

- ใช้เฉพาะข้อมูลจาก evidence
- ระบุ star rating
- ถ้าแนะนำหลายตัวเลือกให้แยกเป็นข้อ
- ห้ามแต่งชื่อร้านหรือข้อมูลที่ไม่มีในรีวิว

### 8.1 การ fine-tune LLM

ในโปรเจกต์นี้เพิ่ม pipeline สำหรับ fine-tune LLM จริงด้วย `QLoRA/LoRA` บน `scb10x/typhoon-7b`

องค์ประกอบหลัก:

- สร้าง SFT dataset จาก query labels + retrieved evidence
- ใช้ `peft` สำหรับ LoRA adapter
- ใช้ `trl` สำหรับ supervised fine-tuning
- ใช้ quantized loading ผ่าน `bitsandbytes`

ไฟล์ที่เกี่ยวข้อง:

- `wongnai_qa/finetuning.py`
- `scripts/build_llm_sft_dataset.py`
- `scripts/train_llm_lora.py`

adapter ที่ train แล้วถูกเก็บไว้ที่:

- `artifacts/typhoon_lora_adapter`

## 9. ประเภทคำถามที่รองรับตามโจทย์

### 9.1 คำถามเกี่ยวกับสัญชาติอาหาร

ตัวอย่าง:

- ร้านอาหารไทยมีที่ไหนน่าสนใจ
- มีร้านอาหารญี่ปุ่นในเชียงใหม่ไหม
- อยากกินอาหารอิตาลีหรือพิซซ่า

### 9.2 คำถามเกี่ยวกับประเภทอาหาร

ตัวอย่าง:

- อยากกินอาหารทะเลสด ๆ
- มีร้านเบเกอรี่หรือคาเฟ่ขนมไหม
- แนะนำร้านก๋วยเตี๋ยวหรืออาหารตามสั่ง

### 9.3 คำถามเกี่ยวกับบรรยากาศและราคา

ตัวอย่าง:

- ร้านบรรยากาศหรูสำหรับเดต
- มีร้านติดแอร์ราคาไม่แพงไหม
- ร้านข้างทางที่คุ้มราคาแนะนำหน่อย

### 9.4 คำถามเกี่ยวกับสถานที่ตั้ง

ตัวอย่าง:

- ร้านติดทะเลแถวพัทยา
- ร้านบรรยากาศสงบในเชียงใหม่
- มีร้านบนเขาหรือวิวดีไหม

### 9.5 คำถามแบบผสม

ตัวอย่าง:

- อาหารทะเลแบบไทย ๆ ติดชายหาดแถวพัทยา
- ร้านญี่ปุ่นคุ้มราคาในเชียงใหม่
- คาเฟ่ขนมบรรยากาศเงียบ ๆ ในกรุงเทพ

## 10. วิธีแสดงผลให้เห็นว่า query ใช้งานได้จริง

ระบบรองรับการแสดงผล 3 แบบ:

- CLI
- FastAPI
- Streamlit UI

โดยในการตอบคำถามหนึ่งครั้ง ระบบจะแสดง:

1. baseline retrieval result
2. finetuned retrieval result
3. generated answer
4. หลักฐานอ้างอิงเป็นรีวิวพร้อม star rating

## 11. จุดเด่นของแนวทางนี้

- ใช้ dataset ตามโจทย์จริง
- ใช้ query labels และ food dictionary จริง
- รองรับหลายภาษาอย่างน้อยไทยและอังกฤษ
- มีทั้ง retrieval และ generative QA
- มี baseline เทียบกับ finetuned
- มี evaluation pipeline

## 12. ข้อจำกัดของงานปัจจุบัน

- finetuned model ยังเป็นการ tune retriever/reranker ไม่ใช่ LLM fine-tuning เต็มรูปแบบ
- relevance สำหรับ evaluation ยังเป็น pseudo-label
- ถ้ารีวิวไม่มีชื่อร้าน ระบบอาจแสดงเป็นตัวเลือกแทนชื่อร้านจริง
- generation model ต้องใช้ GPU

## 13. คำสั่งที่ใช้สาธิตในรายงาน

### รัน backend

```powershell
uv run uvicorn main:app --host 127.0.0.1 --port 8001
```

### รัน frontend

```powershell
uv run streamlit run streamlit_app.py
```

### รัน query ผ่าน CLI

```powershell
uv run python main.py --query "อาหารทะเลแบบไทยๆ ติดชายหาดแถวพัทยา"
```

### tune retriever

```powershell
uv run python scripts/tune_retriever.py
```

### evaluate baseline vs finetuned

```powershell
uv run python scripts/evaluate_models.py
```

### รันชุดคำถามตามโจทย์ 5 หมวด

```powershell
uv run python scripts/run_assignment_demo.py
```

### สร้าง SFT dataset สำหรับ LLM

```powershell
uv run python scripts/build_llm_sft_dataset.py
```

### train LoRA adapter สำหรับ LLM

```powershell
uv run python scripts/train_llm_lora.py
```

## 14. ผลการทดลองเบื้องต้นจากเครื่องที่ใช้พัฒนา

ผลที่บันทึกไว้ในรอบพัฒนานี้รันด้วย `sample_size=20` เพื่อให้จบได้บนเครื่องที่มีข้อจำกัดด้านเวลาและทรัพยากร

### 14.1 ผลการ tune retriever

น้ำหนักที่ถูกเลือกจากการ tune เบื้องต้น:

- rating = `0.05`
- tag = `0.45`
- keyword = `0.33`
- exact = `0.0`

ผล train/eval เบื้องต้น:

- train queries = `2`
- eval queries = `1`
- best train hit_rate_at_k = `0.5`
- best train avg_relevant_ratio_at_k = `0.5`

### 14.2 ผล evaluation เบื้องต้น

จาก benchmark ขนาดเล็กในเครื่องพัฒนา:

- benchmark queries = `1`
- baseline hit_rate_at_k = `0.0`
- baseline avg_relevant_ratio_at_k = `0.0`
- finetuned hit_rate_at_k = `0.0`
- finetuned avg_relevant_ratio_at_k = `0.0`

ข้อสังเกต:

- benchmark ที่ใช้ในรอบนี้เล็กมาก จึงยังสรุปเชิงสถิติไม่ได้
- จุดประสงค์ของผลชุดนี้คือยืนยันว่า pipeline `tune -> evaluate -> compare` ทำงานจริง
- หากจะใช้ส่งอาจารย์ควรเพิ่มจำนวน sample และ benchmark ให้มากขึ้นเมื่อมีเวลารันมากพอ

### 14.3 ผล query ตัวอย่างตามโจทย์

ระบบสามารถรัน query ได้ครบ 5 หมวด:

1. cuisine
2. food_type
3. ambience_price
4. location
5. mixed

ผลลัพธ์ที่ได้แสดงทั้ง:

- baseline answer
- finetuned retrieval answer
- generation status

โดยในรอบทดลองนี้ `generation_status = disabled` เพื่อให้ demo จบได้เร็วบนเครื่องพัฒนา หากต้องการเปิด generation สามารถตั้ง `ENABLE_GENERATION=1`

### 14.4 ผลการ fine-tune LLM

ได้ทดลอง train LoRA adapter บนเครื่องที่มี GPU `NVIDIA GeForce RTX 5070` สำเร็จแล้ว

ผลลัพธ์ที่ได้:

- สร้าง SFT dataset ได้ที่ `artifacts/llm_sft_dataset.jsonl`
- train LoRA adapter สำเร็จที่ `artifacts/typhoon_lora_adapter`
- ระบบ inference สามารถโหลด adapter นี้อัตโนมัติและสร้างคำตอบได้จริง

ตัวอย่างการทดสอบหลัง train:

- query: `อยากได้อาหารทะเลแบบไทยๆ ติดชายหาดแถวพัทยา`
- ระบบสามารถตอบเป็นภาษาไทยและแนะนำหลายตัวเลือกพร้อมลำดับคำตอบได้

ข้อสังเกต:

- การ train รอบนี้เป็นรอบสาธิตขนาดเล็กเพื่อพิสูจน์ pipeline ว่าทำงานจริง
- หากต้องการคุณภาพเชิงโมเดลที่ดีขึ้น ควรเพิ่มขนาด SFT dataset และจำนวน epoch

## 15. สรุป

โปรเจกต์นี้ตอบโจทย์งานในด้านการใช้ข้อมูล Wongnai review dataset เพื่อสร้างระบบถามตอบเกี่ยวกับอาหารและร้านอาหาร โดยใช้ทั้ง retrieval และ generative QA พร้อมทั้งมี baseline และ finetuned model สำหรับเปรียบเทียบ รวมถึงมีขั้นตอน preprocessing, tuning และ evaluation ที่อธิบายได้อย่างเป็นระบบ
