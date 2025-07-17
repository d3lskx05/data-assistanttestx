import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import functools

# ───── Локальная модель (Flan-T5-small) ─────
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# ── Загрузка семантической базы ──
@st.cache_data
def load_data():
    return load_all_excels()

st.set_page_config(page_title="Chat + Semantic Search", layout="wide")
st.title("💬 Чат‑бот + Semantic Search")

# Инициализация
llm = load_llm()
df = load_data()
if "history" not in st.session_state:
    st.session_state.history = []

# Отображение диалога
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# Ввод пользователя
prompt = st.chat_input("Введите сообщение...")

if prompt:
    # Пользователь
    st.session_state.history.append(("user", prompt))
    with st.chat_message("assistant"):
        st.markdown("🤖 Думаю…")

    # Генерация LLM
    response = llm(prompt, max_new_tokens=100)[0]["generated_text"]
    st.session_state.history.pop()  # убираем "Думаю..."
    st.session_state.history.append(("assistant", response))

    # Semantic search
    sem = semantic_search(prompt, df, top_k=3, threshold=0.5)
    kw = keyword_search(prompt, df)

    combined = []
    if sem:
        for s, phrase, topics, comment in sem:
            combined.append(f"🔹 {phrase} (_{', '.join(topics)}_) — {s:.2f}")
    elif kw:
        for phrase, topics, comment in kw:
            combined.append(f"🔸 {phrase} (_{', '.join(topics)}_)")
    else:
        combined.append("⚠️ Ничего не найдено в базе.")

    combined_txt = "\n\n".join(combined)
    st.session_state.history.append(("assistant", combined_txt))

    # Показываем ответы (LLM + база)
    for role, msg in st.session_state.history[-2:]:
        with st.chat_message(role):
            st.markdown(msg)
