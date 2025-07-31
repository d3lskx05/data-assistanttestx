# app.py
import streamlit as st
import pandas as pd
import os
from utils import load_data, load_model, keyword_search, semantic_search, log_query

st.title("Поиск по фразам")

# Загрузка данных
df = load_data()

# Фильтрация по тематикам
if "theme" in df.columns:
    themes = sorted(df["theme"].dropna().unique().tolist())
    selected_themes = st.multiselect("Выберите тематику", options=themes)
    if selected_themes:
        df_filtered = df[df["theme"].isin(selected_themes)]
    else:
        df_filtered = df
else:
    df_filtered = df

# Выбор режима поиска
mode = st.selectbox("Тип поиска", ("Ключевой", "Семантический"))

# Ввод запроса
query = st.text_input("Введите фразу для поиска")

# Кнопка поиска
if st.button("Поиск"):
    if not query:
        st.warning("Пожалуйста, введите фразу для поиска.")
    else:
        results_df = pd.DataFrame()
        # Разделение запроса по слешу, если есть
        parts = [part.strip() for part in query.split("/") if part.strip()]
        # Выполнение поиска для каждой части
        if mode == "Ключевой":
            for part in parts:
                res = keyword_search(df_filtered, part)
                results_df = pd.concat([results_df, res], ignore_index=True)
        else:  # Семантический поиск
            model = load_model()
            for part in parts:
                res = semantic_search(df_filtered, part, model)
                results_df = pd.concat([results_df, res], ignore_index=True)
        # Удаление дубликатов
        if not results_df.empty:
            if "phrase" in results_df.columns:
                results_df = results_df.drop_duplicates(subset=["phrase"])
        # Вывод результатов
        if not results_df.empty:
            st.write("Результаты поиска:")
            cols = ["phrase"]
            if "theme" in results_df.columns:
                cols.append("theme")
            if "comment" in results_df.columns:
                cols.append("comment")
            st.dataframe(results_df[cols])
            log_query(query, "SUCCESS")
        else:
            st.write("Ничего не найдено.")
            log_query(query, "NO_MATCH")

# Кнопка скачивания логов
log_path = "/tmp/query_log.txt"
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        log_data = f.read()
    st.download_button("Скачать логи", log_data, file_name="query_log.txt", mime="text/plain")
