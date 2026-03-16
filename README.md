# Wongnai Restaurant QA

ระบบถามตอบเกี่ยวกับอาหารและร้านอาหารจากรีวิว Wongnai โดยใช้ทั้ง retrieval และ generative QA พร้อมส่วนเปรียบเทียบระหว่าง `baseline model` และ `finetuned model`

## องค์ประกอบของระบบ

- `FastAPI` backend สำหรับรับ query และคืนผลลัพธ์
- `Streamlit` frontend สำหรับสาธิตการใช้งาน
- shared service layer ที่ใช้ร่วมกันระหว่าง CLI, API และ UI
- retriever แบบ baseline
- retriever/reranker แบบ finetuned จาก labeled queries
- LLM สำหรับสรุปคำตอบภาษาไทย
- QLoRA/LoRA pipeline สำหรับ fine-tune LLM บน GPU

## โครงสร้างหลักของโปรเจกต์

- [wongnai_qa/config.py](c:/Users/com/Desktop/NLP/wongnai_qa/config.py): ค่าตั้งต้นของระบบ, path และชุดคำถามตัวอย่าง
- [wongnai_qa/preprocessing.py](c:/Users/com/Desktop/NLP/wongnai_qa/preprocessing.py): preprocessing ข้อมูล, tagging metadata และวิเคราะห์ query
- [wongnai_qa/retrieval.py](c:/Users/com/Desktop/NLP/wongnai_qa/retrieval.py): embeddings, vector store, baseline retrieval, finetuned retrieval และ scoring weights
- [wongnai_qa/generation.py](c:/Users/com/Desktop/NLP/wongnai_qa/generation.py): baseline summary และ prompt สำหรับ generative QA
- [wongnai_qa/llm.py](c:/Users/com/Desktop/NLP/wongnai_qa/llm.py): โหลด LLM
- [wongnai_qa/service.py](c:/Users/com/Desktop/NLP/wongnai_qa/service.py): orchestration layer หลัก
- [wongnai_qa/api.py](c:/Users/com/Desktop/NLP/wongnai_qa/api.py): FastAPI backend
- [wongnai_qa/ui.py](c:/Users/com/Desktop/NLP/wongnai_qa/ui.py): logic ของ Streamlit UI
- [wongnai_qa/evaluation.py](c:/Users/com/Desktop/NLP/wongnai_qa/evaluation.py): benchmark, tuning และ evaluation
- [wongnai_qa/finetuning.py](c:/Users/com/Desktop/NLP/wongnai_qa/finetuning.py): สร้าง SFT dataset และ train LoRA adapter
- [scripts/tune_retriever.py](c:/Users/com/Desktop/NLP/scripts/tune_retriever.py): tune น้ำหนักของ finetuned retriever
- [scripts/evaluate_models.py](c:/Users/com/Desktop/NLP/scripts/evaluate_models.py): วัดผล baseline เทียบ finetuned
- [scripts/run_assignment_demo.py](c:/Users/com/Desktop/NLP/scripts/run_assignment_demo.py): รัน query ตัวอย่างครบ 5 หมวดตามโจทย์
- [scripts/build_llm_sft_dataset.py](c:/Users/com/Desktop/NLP/scripts/build_llm_sft_dataset.py): สร้าง supervised fine-tuning dataset สำหรับ LLM
- [scripts/train_llm_lora.py](c:/Users/com/Desktop/NLP/scripts/train_llm_lora.py): train LoRA adapter สำหรับ `typhoon-7b`

## โครงสร้างโฟลเดอร์

```text
NLP/
|- wongnai_qa/        # package หลักของระบบ
|- review_dataset/    # dataset และ dictionary
|- scripts/           # สคริปต์สำหรับ tune, evaluate, demo
|- docs/              # เอกสารอธิบายระบบและรายงาน
|- chroma_db/         # vector index ที่ persist ไว้
|- artifacts/         # น้ำหนักหรือผลลัพธ์ที่สร้างเพิ่มจากการ tune
|- main.py            # entrypoint สำหรับ CLI
|- streamlit_app.py   # entrypoint สำหรับ Streamlit
|- api_server.py      # entrypoint สำหรับ FastAPI
```

## เอกสารประกอบ

- [docs/system_workflow.md](c:/Users/com/Desktop/NLP/docs/system_workflow.md): อธิบายการทำงานของระบบแบบละเอียด
- [docs/assignment_report_th.md](c:/Users/com/Desktop/NLP/docs/assignment_report_th.md): เอกสารสรุปแนวคิดสำหรับส่งงานอาจารย์

## การติดตั้ง

```powershell
uv sync
```

หรือถ้าใช้ `requirements.txt`

```powershell
pip install -r requirements.txt
```

## การรัน Backend

```powershell
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

## การรัน Frontend

เปิดอีก terminal แล้วรัน:

```powershell
uv run streamlit run streamlit_app.py
```

ถ้า API ไม่ได้รันอยู่ที่ host หรือ port เดียวกัน:

```powershell
$env:WONGNAI_API_URL="http://127.0.0.1:8000"
uv run streamlit run streamlit_app.py
```

## การทดลองผ่าน CLI

```powershell
uv run python main.py --query "อาหารทะเลแบบไทยๆ ติดชายหาดแถวพัทยา"
```

ตัวอย่างการทดสอบแบบ retrieval-only:

```powershell
uv run python main.py --sample-size 300 --rebuild --skip-llm --query "ร้านติดทะเลพัทยา"
```

## การ tune retriever

สร้างน้ำหนักของ finetuned retriever จาก query labels:

```powershell
uv run python scripts/tune_retriever.py
```

ไฟล์น้ำหนักที่ tune แล้วจะถูกบันทึกไว้ที่:

- `artifacts/tuned_retriever_weights.json`

## การวัดผล baseline เทียบกับ finetuned

```powershell
uv run python scripts/evaluate_models.py
```

ผลที่ได้จะเป็น metrics สำหรับเปรียบเทียบอย่างน้อย:

- `hit_rate_at_k`
- `avg_relevant_ratio_at_k`

## การ fine-tune LLM ด้วย LoRA/QLoRA

สร้าง SFT dataset:

```powershell
uv run python scripts/build_llm_sft_dataset.py
```

train LoRA adapter:

```powershell
uv run python scripts/train_llm_lora.py
```

adapter ที่ train แล้วจะถูกเก็บไว้ที่:

- `artifacts/typhoon_lora_adapter`

ถ้ามี adapter อยู่แล้ว ระบบ inference จะโหลด adapter นี้ให้อัตโนมัติ

## การรันชุด query ตามโจทย์ 5 หมวด

```powershell
uv run python scripts/run_assignment_demo.py
```

ถ้าต้องการเปิด generative answer ใน demo:

```powershell
$env:ENABLE_GENERATION="1"
uv run python scripts/run_assignment_demo.py
```

สคริปต์นี้จะรัน query ครบหมวด:

1. สัญชาติอาหาร
2. ประเภทอาหาร
3. บรรยากาศและราคา
4. สถานที่ตั้ง
5. คำถามแบบผสม

## หมายเหตุสำคัญ

- embedding model เริ่มต้น: `intfloat/multilingual-e5-large`
- generator model เริ่มต้น: `scb10x/typhoon-7b`
- โปรเจกต์นี้ตั้งค่าให้รองรับการใช้งานแบบ cached / offline สำหรับ Hugging Face model เป็นหลัก
- baseline model ในโปรเจกต์นี้คือ pure vector retrieval
- finetuned model ในโปรเจกต์นี้คือ retriever/reranker ที่ปรับน้ำหนักจาก labeled queries ใน dataset
- generative answer จะอิงเอกสารที่ได้จากฝั่ง finetuned retriever
- ถ้า train LoRA adapter แล้ว `load_llm()` จะใช้โมเดลฐานร่วมกับ adapter ที่ fine-tune ไว้
