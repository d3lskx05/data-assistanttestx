# app.py

import streamlit as st
import numpy as np
from utils import load_model, load_data, build_faiss_index, keyword_search, semantic_search

# Настройка заголовка приложения
st.title("Поиск по базе фраз")

# 1. Загрузка модели и данных (с помощью кэширования для скорости)
@st.cache(allow_output_mutation=True)
def prepare_data(model, urls):
    df = load_data(urls)
    # Извлекаем матрицу эмбеддингов из столбца 'embeddings'
    if 'embeddings' in df.columns and len(df) > 0:
        embeddings = np.vstack(df['embeddings'].values)
        index = build_faiss_index(embeddings)
    else:
        embeddings = np.array([])
        index = None
    return df, embeddings, index

model = load_model()
# Список URL на GitHub (CSV/XLSX) – пример
urls = [
    "https://raw.githubusercontent.com/youruser/yourrepo/main/data1.csv",
    "https://raw.githubusercontent.com/youruser/yourrepo/main/data2.xlsx"
]
df, embeddings, index = prepare_data(model, urls)

# 2. Фильтр по темам
all_topics = sorted(df['topics'].unique())
selected_topics = st.multiselect(
    "Темы (для поиска можно начать ввод темы и выбрать вариант)", 
    options=all_topics
)
# Фильтрация DataFrame по выбранным темам
if selected_topics:
    df_filtered = df[df['topics'].isin(selected_topics)]
else:
    df_filtered = df.copy()

st.write(f"Найдено фраз: {len(df_filtered)} по выбранным темам")

# 3. Поле ввода запроса и параметров поиска
query = st.text_input("Введите запрос для поиска:")
threshold = st.slider("Порог схожести (threshold)", 0.0, 1.0, 0.5, 0.01)
top_k = st.number_input("Top K результатов", min_value=1, max_value=100, value=5, step=1)

# 4. Если запрос не пустой, выполняем поиск
if query:
    # Список индексов, ограниченный выбранными темами
    if selected_topics:
        # Чтобы поисковый индекс учитывал фильтрацию, можно построить новый индекс на df_filtered.embeddings
        emb_filtered = np.vstack(df_filtered['embeddings'].values) if len(df_filtered) > 0 else np.array([])
        index_filtered = build_faiss_index(emb_filtered) if len(emb_filtered) > 0 else None
        # Semantic Search
        sem_results = semantic_search(query, df_filtered, index_filtered, embeddings, model, threshold, top_k)
        # Keyword Search
        key_results = keyword_search(query, df_filtered)
    else:
        sem_results = semantic_search(query, df_filtered, index, embeddings, model, threshold, top_k)
        key_results = keyword_search(query, df_filtered)
    
    # 5. Вывод результатов
    st.subheader("Результаты семантического поиска")
    if not sem_results.empty:
        for _, row in sem_results.iterrows():
            st.markdown(f"**Фраза:** {row['phrase_full']}")
            st.markdown(f"*Комментарий:* {row['comment']}")
            st.write("")  # пустая строка между карточками
    else:
        st.write("По вашему запросу в семантическом поиске ничего не найдено.")
    
    st.subheader("Результаты точного поиска (по словам)")
    if not key_results.empty:
        for _, row in key_results.iterrows():
            st.markdown(f"**Фраза:** {row['phrase_full']}")
            st.markdown(f"*Комментарий:* {row['comment']}")
            st.write("")
    else:
        st.write("По вашему запросу в точном поиске ничего не найдено.")
