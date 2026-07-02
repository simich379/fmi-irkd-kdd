import tempfile
import time
from pathlib import Path

import streamlit as st

from agent_core import (
    extract_text_from_file, chunk_text, LocalVectorStore,
    answer_with_rag, ollama_vision
)

st.set_page_config(
    page_title="Локален мултимодален AI агент",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Локален мултимодален AI агент: Vision + Tool Use + RAG")
st.caption("Демо за локална обработка на PDF/TXT/изображения, semantic search и оценяване на системата")

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
TEXT_EXTENSIONS = [".txt", ".pdf"]


def get_file_type(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    return "unknown"


def get_vision_prompt() -> str:
    return """
You are a local vision model used in a university demo.

Answer ONLY in Bulgarian.
Do not use Russian, Serbian, Latin transliteration or mixed languages.
Do not invent numbers, dates, prices or names.
If the text is not readable, say exactly: "Текстът не е достатъчно четим от изображението."
Describe only what is clearly visible.
Keep the answer short and structured.

Task:
Describe the image briefly in Bulgarian.
"""


with st.sidebar:
    st.header("Конфигурация")
    chunk_size = st.slider("Chunk size", 200, 1800, 800, step=100)
    overlap = st.slider("Overlap", 0, 300, 120, step=20)
    top_k = st.slider("Top-k", 1, 8, 3)
    llm_model = st.text_input("LLM модел", "llama3.2")
    vision_model = st.text_input("Vision модел", "llava")

uploaded = st.file_uploader(
    "Качи файл",
    type=["pdf", "txt", "png", "jpg", "jpeg", "webp"]
)

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    file_type = get_file_type(suffix)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.getbuffer())
        temp_path = f.name

    st.subheader("1. File Processing / Tool Use")
    st.write(f"Качен файл: **{uploaded.name}**")
    st.write(f"Тип файл: **{file_type}**")

    t0 = time.time()
    text, tool = extract_text_from_file(temp_path)
    extraction_time = time.time() - t0

    st.write(f"Използван tool: **{tool}**")
    st.write(f"Време за извличане: **{extraction_time:.2f} сек.**")

    if file_type == "image":
        col1, col2 = st.columns(2)

        with col1:
            st.image(temp_path, caption="Качено изображение", use_container_width=True)

        with col2:
            st.write("Vision analysis:")
            vision_prompt = get_vision_prompt()

            try:
                vision_result = ollama_vision(temp_path, vision_prompt, model=vision_model)
            except Exception as e:
                vision_result = f"Vision анализът не можа да се изпълни. Грешка: {e}"

            st.info(vision_result)

        st.write("OCR result:")
        st.success(text[:1000] if text else "OCR не извлече текст.")

    elif file_type == "text":
        st.info(
            "Файлът е текстов/PDF документ. Vision модел не се използва. "
            "За този тип файл се използва file parsing + chunking + semantic search + RAG."
        )

    else:
        st.warning("Неподдържан тип файл.")

    st.text_area("Извлечен текст", text[:4000] if text else "", height=180)

    if not text or not text.strip():
        st.warning(
            "Няма извлечен текст от файла. "
            "Ако файлът е изображение, провери Tesseract OCR. "
            "Ако е PDF, провери дали съдържа selectable text или дали е сканиран PDF."
        )
        st.stop()

    st.subheader("2. Chunking + Semantic Search")
    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
        source=uploaded.name
    )

    st.write(f"Брой chunks: **{len(chunks)}**")

    store = LocalVectorStore()

    if chunks:
        store.build(chunks)
    else:
        st.warning("Не са създадени chunks. Провери chunk size и съдържанието на файла.")
        st.stop()

    query = st.text_input("Въпрос към документа", "Какви са основните идеи в документа?")

    if st.button("Задай въпрос"):
        t1 = time.time()
        retrieved = store.search(query, top_k=top_k)
        retrieval_time = time.time() - t1

        t2 = time.time()
        answer = answer_with_rag(query, retrieved, model=llm_model)
        generation_time = time.time() - t2

        st.subheader("3. Retrieval резултати")

        if retrieved:
            for r in retrieved:
                with st.expander(f"{r['chunk_id']} | score={r['score']:.3f}"):
                    st.write(r["text"])
        else:
            st.warning("Няма намерени релевантни chunks.")

        st.subheader("4. Финален отговор")
        st.success(answer)

        st.subheader("5. Измервания")
        st.table({
            "metric": [
                "file_type",
                "extraction_tool",
                "extraction_time",
                "retrieval_time",
                "generation_time",
                "total_time",
                "chunks",
                "top_k"
            ],
            "value": [
                file_type,
                tool,
                round(extraction_time, 2),
                round(retrieval_time, 2),
                round(generation_time, 2),
                round(extraction_time + retrieval_time + generation_time, 2),
                len(chunks),
                top_k
            ]
        })

else:
    st.info("Качи файл от папка tests/test_files, за да стартираш демото.")
