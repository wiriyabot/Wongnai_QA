from __future__ import annotations

import os
from html import escape
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("WONGNAI_API_URL", "http://127.0.0.1:8001")
DEFAULT_TOP_K = 4
DEFAULT_FETCH_K = 12
DEFAULT_SAMPLE_SIZE = 1000


def configure_page() -> None:
    st.set_page_config(
        page_title="Wongnai Restaurant QA",
        page_icon="🍜",
        layout="wide",
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');
        :root {
            --bg-1: #f6efe4;
            --bg-2: #fcfaf5;
            --bg-3: #eef4f8;
            --ink-1: #102033;
            --ink-2: #566476;
            --line: rgba(16, 32, 51, 0.1);
            --card: rgba(255, 255, 255, 0.84);
        }
        * {
            box-sizing: border-box;
        }
        html, body, [class*="css"] {
            font-family: "IBM Plex Sans Thai", sans-serif;
            color: var(--ink-1);
        }
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(239, 124, 69, 0.2), transparent 26%),
                radial-gradient(circle at 100% 0%, rgba(31, 111, 120, 0.16), transparent 24%),
                radial-gradient(circle at 50% 100%, rgba(220, 180, 120, 0.12), transparent 20%),
                linear-gradient(145deg, var(--bg-1) 0%, var(--bg-2) 46%, var(--bg-3) 100%);
        }
        .block-container {
            padding-top: 2.8rem;
            padding-bottom: 2.2rem;
            max-width: 1280px;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }
        [data-testid="stSpinner"] > div {
            flex-wrap: wrap;
            row-gap: 0.45rem;
            max-width: 100%;
        }
        [data-testid="stSpinner"] p {
            margin: 0;
            overflow-wrap: anywhere;
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2rem 1.7rem;
            border-radius: 34px;
            background:
                radial-gradient(circle at top right, rgba(255, 216, 145, 0.2), transparent 24%),
                linear-gradient(135deg, rgba(13, 27, 42, 0.98), rgba(30, 60, 84, 0.94) 58%, rgba(39, 92, 84, 0.9));
            color: #fff9ef;
            box-shadow: 0 30px 80px rgba(22, 37, 55, 0.22);
            margin-top: 0.6rem;
            margin-bottom: 1.4rem;
        }
        .hero::after {
            content: "";
            position: absolute;
            inset: auto -8% -26% auto;
            width: 20rem;
            height: 20rem;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.15), transparent 64%);
        }
        .hero h1 {
            margin: 0 0 0.65rem;
            max-width: 12ch;
            font-family: "Space Grotesk", sans-serif;
            font-size: 3rem;
            line-height: 0.98;
            color: #fff9ef !important;
        }
        .hero p {
            margin: 0;
            max-width: 48rem;
            color: rgba(255, 248, 239, 0.92) !important;
            line-height: 1.65;
        }
        .hero,
        .hero * {
            color: #fff9ef !important;
        }
        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.25rem;
        }
        .hero-badge {
            padding: 0.55rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.08);
            color: #fffdf7;
            font-size: 0.82rem;
        }
        .panel {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.15rem 1.15rem 1.2rem;
            box-shadow: 0 18px 55px rgba(22, 36, 52, 0.08);
            backdrop-filter: blur(14px);
            overflow: hidden;
        }
        .query-panel {
            max-width: 880px;
            margin: 0 auto 1rem;
        }
        .results-shell {
            max-width: 1120px;
            margin: 0 auto;
        }
        .panel-title {
            margin-bottom: 0.95rem;
        }
        .eyebrow {
            display: inline-block;
            margin-bottom: 0.35rem;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: rgba(16, 32, 51, 0.06);
            color: var(--ink-2);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .panel-title h3 {
            margin: 0;
            font-size: 1.2rem;
        }
        .panel-title p {
            margin: 0.25rem 0 0;
            color: var(--ink-2);
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .stButton > button {
            width: 100%;
            min-height: 3.1rem;
            border-radius: 18px;
            border: none;
            background: linear-gradient(135deg, #ef7c45, #e1593f);
            color: white;
            font-weight: 700;
            box-shadow: 0 16px 32px rgba(225, 89, 63, 0.24);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #f08c54, #df5234);
        }
        .stSelectbox label, .stTextArea label {
            font-weight: 600;
            color: var(--ink-1);
        }
        .stTextArea textarea {
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.92);
            font-size: 0.95rem;
        }
        .hint-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.9rem;
        }
        .hint-pill {
            padding: 0.42rem 0.68rem;
            border-radius: 999px;
            background: rgba(16, 32, 51, 0.06);
            color: var(--ink-2);
            font-size: 0.8rem;
            font-weight: 600;
        }
        .metric-chip {
            display: inline-block;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            padding: 0.45rem 0.72rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(16, 32, 51, 0.96), rgba(36, 66, 92, 0.94));
            color: #fff9f1;
            font-size: 0.84rem;
        }
        .section-heading {
            margin: 1.6rem 0 0.8rem;
        }
        .section-heading h2 {
            margin: 0;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.55rem;
        }
        .section-heading p {
            margin: 0.2rem 0 0;
            color: var(--ink-2);
        }
        .answer-card {
            border-radius: 28px;
            padding: 1.1rem 1.15rem;
            min-height: 14rem;
            box-shadow: 0 18px 45px rgba(22, 36, 52, 0.08);
        }
        .improved {
            background: linear-gradient(180deg, rgba(234,246,246,0.98), rgba(246,251,251,0.96));
            border: 1px solid rgba(31, 111, 120, 0.16);
        }
        .small-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--ink-2);
        }
        .answer-title {
            margin: 0.15rem 0 0.7rem;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.25rem;
            color: var(--ink-1);
        }
        .answer-body {
            white-space: pre-wrap;
            font-family: "IBM Plex Sans Thai", sans-serif;
            line-height: 1.72;
            color: #203145;
            margin: 0;
        }
        .doc-card {
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(244,247,251,0.92));
            border: 1px solid rgba(24, 38, 58, 0.08);
            padding: 1rem 1rem 0.95rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 16px 40px rgba(22, 36, 52, 0.06);
        }
        .doc-tags {
            margin: 0.7rem 0 0.9rem;
        }
        .doc-tag {
            display: inline-block;
            margin: 0 0.42rem 0.42rem 0;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            background: rgba(31, 111, 120, 0.1);
            color: #154a51;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .doc-body {
            color: #243549;
            line-height: 1.7;
        }
        @media (max-width: 900px) {
            .hero h1 {
                font-size: 2.25rem;
            }
            .query-panel, .results-shell {
                max-width: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_get(path: str) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def render_tags(query_profile: dict[str, Any]) -> None:
    chips = []
    for group_name, values in query_profile["detected_tags"].items():
        for value in values:
            chips.append(f"{group_name}: {value}")
    for term in query_profile.get("query_terms", []):
        chips.append(f"term: {term}")
    if not chips:
        st.caption("No explicit tags detected")
        return
    st.markdown("".join(f"<span class='metric-chip'>{escape(chip)}</span>" for chip in chips), unsafe_allow_html=True)


def render_retrieved_documents(documents: list[dict[str, Any]]) -> None:
    for index, document in enumerate(documents, start=1):
        metadata = document["metadata"]
        tag_fragments = []
        for key in ["cuisine", "food_type", "ambience", "price", "location"]:
            value = str(metadata.get(key, "")).strip("|")
            if value:
                tag_fragments.append(f"<span class='doc-tag'>{escape(key)}: {escape(value.replace('|', ', '))}</span>")
        st.markdown(
            f"""
            <div class="doc-card">
                <div class="small-label">Retrieved Review {index}</div>
                <div class="answer-title">Rating {escape(str(metadata.get("rating", "N/A")))} / 5</div>
                <div class="doc-tags">{"".join(tag_fragments) if tag_fragments else "<span class='doc-tag'>No explicit tags</span>"}</div>
                <div class="doc-body">{escape(str(document["page_content"]))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_section_heading(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{escape(title)}</h2>
            <p>{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="panel results-shell" style="text-align:center; padding: 1.6rem 1.2rem;">
            <span class="eyebrow">Ready</span>
            <h3 style="margin: 0.2rem 0 0.45rem;">พิมพ์คำถามเกี่ยวกับร้านอาหาร แล้วกดค้นหา</h3>
            <p style="margin: 0; color: var(--ink-2);">
                ระบบจะแสดงคำตอบหลักพร้อมรีวิวที่ใช้ประกอบคำตอบโดยอัตโนมัติ
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Wongnai Restaurant QA</h1>
            <p>ถามหาร้านอาหารจากรีวิวได้ตรง ๆ ระบบจะสรุปคำตอบที่อ่านง่ายและแสดงรีวิวที่เกี่ยวข้องให้ทันที</p>
            <div class="hero-badges">
                <span class="hero-badge">ค้นหาจากรีวิวจริง</span>
                <span class="hero-badge">ตอบเป็นภาษาคนอ่านง่าย</span>
                <span class="hero-badge">มีหลักฐานอ้างอิงประกอบ</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='panel query-panel'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-title">
            <span class="eyebrow">Ask</span>
            <h3>อยากกินอะไร หรืออยากได้ร้านแบบไหน</h3>
            <p>พิมพ์คำถามแบบภาษาคนทั่วไปได้เลย เช่น อาหารทะเลริมทะเล คาเฟ่ถ่ายรูปสวย หรือร้านญี่ปุ่นคุ้มราคา</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        demo_queries = api_get("/demo-queries")["queries"]
    except Exception:
        demo_queries = []

    selected_demo = st.selectbox(
        "ตัวอย่างคำถาม",
        [""] + demo_queries,
        help="เลือกคำถามตัวอย่าง หรือพิมพ์ใหม่เองด้านล่าง",
    )
    default_query = selected_demo or "อาหารทะเลแบบไทยๆ ติดชายหาดแถวพัทยา"
    query = st.text_area("คำถาม", value=default_query, height=120, label_visibility="collapsed")
    st.markdown(
        """
        <div class="hint-row">
            <span class="hint-pill">ร้านอาหารทะเล</span>
            <span class="hint-pill">คาเฟ่บรรยากาศดี</span>
            <span class="hint-pill">ร้านญี่ปุ่นคุ้มราคา</span>
            <span class="hint-pill">เหมาะกับครอบครัว</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    submitted = st.button("ค้นหาร้าน", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        render_empty_state()
        return

    payload = {
        "query": query,
        "top_k": DEFAULT_TOP_K,
        "fetch_k": DEFAULT_FETCH_K,
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "include_improved": True,
        "rebuild": False,
    }

    with st.spinner("กำลังค้นหาและสรุปคำตอบ..."):
        result = api_post("/query", payload)

    st.markdown("<div class='results-shell'>", unsafe_allow_html=True)
    render_section_heading("Detected Query Signals", "The parser extracts explicit terms and semantic hints from your prompt.")
    render_tags(result["query_profile"])

    render_section_heading("เปรียบเทียบโมเดล", "แสดงผล retrieval ของ baseline model เทียบกับ finetuned retriever")
    baseline_col, finetuned_col = st.columns(2, gap="large")
    with baseline_col:
        st.markdown(
            f"""
            <div class="answer-card improved">
                <div class="small-label">Baseline</div>
                <div class="answer-title">ผลลัพธ์จาก baseline retrieval</div>
                <pre class="answer-body">{escape(str(result["baseline_answer"]))}</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with finetuned_col:
        st.markdown(
            f"""
            <div class="answer-card improved">
                <div class="small-label">Finetuned</div>
                <div class="answer-title">ผลลัพธ์จาก finetuned retriever</div>
                <pre class="answer-body">{escape(str(result["finetuned_answer"]))}</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section_heading("คำตอบแบบสรุป", "คำตอบภาษาไทยที่สร้างจากเอกสารฝั่ง finetuned retriever")
    improved_answer = result["improved_answer"] or result["finetuned_answer"]
    st.markdown(
        f"""
        <div class="answer-card improved">
            <div class="small-label">Generative QA</div>
            <div class="answer-title">คำตอบที่ระบบสรุปให้ผู้ใช้</div>
            <pre class="answer-body">{escape(str(improved_answer))}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_heading("รีวิวที่เกี่ยวข้อง", "หลักฐานอ้างอิงจากฝั่ง finetuned retriever ที่ใช้ประกอบคำตอบ")
    render_retrieved_documents(result["retrieved_documents"])
    st.markdown("</div>", unsafe_allow_html=True)
