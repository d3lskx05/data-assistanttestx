import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search
import requests

# Загрузка базы
@st.cache_data
def load_data():
    return load_all_excels()

st.set_page_config(page_title="💬 GPT-like бот", layout="wide")
st.title("🤖 GPT-подобный бот + Семантический поиск")

df = load_data()

# Диалоговая история
if "history" not in st.session_state:
    st.session_state.history = []

# Показ истории
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# Ввод пользователя
prompt = st.chat_input("Введите сообщение…")

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("assistant"):
        st.markdown("🧠 Думаю…")

    # Формируем историю для модели
    context = ""
    for role, msg in st.session_state.history:
        prefix = "User:" if role == "user" else "Assistant:"
        context += f"{prefix} {msg}\n"
    context += "Assistant:"

    # Отправка на llama2 через Replicate демо-интерфейс (можно заменить)
    res = requests.post(
        "https://replicate-api-proxy.glitch.me/llama2-7b-chat",
        json={"prompt": context, "temperature": 0.7, "max_new_tokens": 200}
    )

    if res.status_code == 200:
        answer = res.json().get("text", "").strip()
    else:
        answer = "⚠️ Ошибка при обращении к LLM."

    st.session_state.history.append(("assistant", answer))

    # Semantic Search
    sem = semantic_search(prompt, df)
    kw = keyword_search(prompt, df)

    output = []
    if sem:
        for s, phrase, topics, comment in sem:
            output.append(f"🔹 {phrase} (_{', '.join(topics)}_) — {s:.2f}")
    elif kw:
        for phrase, topics, comment in kw:
            output.append(f"🔸 {phrase} (_{', '.join(topics)}_)")
    else:
        output.append("⚠️ Ничего не найдено в базе.")

    st.session_state.history.append(("assistant", "\n".join(output)))

    # Отображение последних 2х реплик
    for role, msg in st.session_state.history[-2:]:
        with st.chat_message(role):
            st.markdown(msg)
