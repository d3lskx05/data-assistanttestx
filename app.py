import streamlit as st
from utils import load_all_files, semantic_search, keyword_search, filter_by_topics

# ----------- Загрузка данных -----------
@st.cache_resource
def load_data():
    return load_all_files()

df = load_data()

# ----------- Интерфейс приложения -----------
st.set_page_config(page_title="Semantic Assistant", layout="wide")
st.title("🔍 Semantic Assistant")

# Блок фильтров
all_topics = sorted({topic for topics in df["topics"] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам", all_topics)

# Выбор режима поиска
search_mode = st.radio("Выберите режим поиска:", ["Точный поиск", "Умный поиск"], horizontal=True)

# Поле поиска
query = st.text_input("Введите поисковый запрос")

# Количество результатов
top_k = st.slider("Количество результатов (для умного поиска)", min_value=1, max_value=50, value=10)

if st.button("Найти"):
    if not query.strip():
        st.warning("Введите поисковый запрос")
    else:
        if search_mode == "Точный поиск":
            results = keyword_search(query, df)
        else:
            results = semantic_search(query, df, top_k=top_k)

        # Фильтрация по тематикам
        results = filter_by_topics(results, selected_topics)

        if not results:
            st.info("Ничего не найдено.")
        else:
            for item in results:
                if len(item) == 4:  # Умный поиск
                    score, phrase, topics, comment = item
                    st.markdown(f"**Фраза:** {phrase}  \n**Схожесть:** {score:.4f}  \n**Тематики:** {', '.join(topics)}  \n**Комментарий:** {comment}")
                else:  # Точный поиск
                    phrase, topics, comment = item
                    st.markdown(f"**Фраза:** {phrase}  \n**Тематики:** {', '.join(topics)}  \n**Комментарий:** {comment}")

            st.success(f"Найдено {len(results)} результатов")
