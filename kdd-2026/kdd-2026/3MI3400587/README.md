# Локален мултимодален AI агент за откриване на знания от документи и изображения чрез Tool Use и RAG

Проект за дисциплината **„Откриване на знания в данни“**.

Изготвил: **Симона Илиева, ф.н. 3MI3400587, специалност ИИОЗ**

## 1. Идея

Проектът реализира локален AI агент, който обработва PDF/TXT/изображения, извлича текст, разделя го на chunks, създава embeddings, извършва semantic search и генерира отговор чрез RAG. При нужда агентът използва инструменти: OCR, file parsing, vector search и Python анализ.

## 2. Архитектура

```text
User Upload: PDF / Image / TXT
        ↓
File Processing: OCR / Parsing / Chunking
        ↓
Embedding Model + Vision Model
        ↓
Vector Store + AI Agent
        ↓
Tool Use + Reasoning + RAG Answer
```

## 3. Технологии

- Python
- Streamlit UI
- Ollama runtime
- LLaVA / llama3.2-vision за Vision анализ
- nomic-embed-text за embeddings
- pytesseract за OCR
- ChromaDB или локален TF-IDF fallback за semantic search
- pandas/scikit-learn за оценяване

## 4. Стартиране

### 4.1. Инсталиране

```bash
cd local_multimodal_ai_agent_project
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 4.2. Инсталиране на Ollama модели

```bash
ollama pull llava
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 4.3. Стартиране на демото

```bash
streamlit run app/streamlit_app.py
```

## 5. Какво да демонстрираш

1. Качваш PDF, TXT или изображение от `tests/test_files`.
2. Системата извлича текст чрез parsing/OCR.
3. Текстът се разделя на chunks.
4. Създават се embeddings или TF-IDF индекси.
5. Задаваш въпрос към документа.
6. Агентът извиква нужния tool и връща отговор.
7. Стартираш оценка с `python evaluation/evaluate_demo.py`.
8. Показваш таблицата с резултати и изводите.

## 6. Оценяване

Оценяват се:

- Response time
- Precision@k
- Recall@k
- Tool Success Rate
- Human Evaluation Score
- OCR качество
- Ефект от chunk size и top-k върху резултатите

## 7. Готови артефакти

- `presentation/AI_Agent_Demo_Presentation.pptx` - презентация
- `evaluation/evaluation_results.xlsx` - таблици с измервания и конфигурации
- `docs/demo_script.md` - текст за представяне
- `docs/evaluation_protocol.md` - методика за оценяване
- `tests/test_plan.csv` - тестови сценарии
