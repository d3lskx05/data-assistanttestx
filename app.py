import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search, get_model
import datetime
import pandas as pd
import os
import csv
import torch  # <-- используется для нарезки тензора эмбеддингов

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")
st.title("🤖 Проверка фраз")

LOG_FILE = "query_log.csv"

# 🔧 Логирование
def log_query(query, semantic_count, keyword_count, status):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["time", "query", "semantic_results", "keyword_results", "status"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query.strip(),
            semantic_count,
            keyword_count,
            status
        ])

@st.cache_data
def get_data():
    df = load_all_excels()
    model = get_model()
    # рассчитываем эмбеддинги для полной таблицы и сохраняем в attrs
    df.attrs['phrase_embs'] = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)
    return df

df = get_data()

# 🔘 Все уникальные тематики
all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам (независимо от поиска):", all_topics)
filter_search_by_topics = st.checkbox("Искать только в выбранных тематиках", value=False)

# 📂 Фразы по выбранным тематикам
if selected_topics:
    st.markdown("### 📂 Фразы по выбранным тематикам:")
    filtered_df = df[df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))]
    for row in filtered_df.itertuples():
        with st.container():
            st.markdown(
                f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                    <div style="font-size: 18px; font-weight: 600; color: #333;">📝 {row.phrase_full}</div>
                    <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(row.topics)}</strong></div>
                </div>""",
                unsafe_allow_html=True
            )
            if row.comment and str(row.comment).strip().lower() != "nan":
                with st.expander("💬 Комментарий", expanded=False):
                    st.markdown(row.comment)

# 📥 Поисковый запрос
query = st.text_input("Введите ваш запрос:")

if query:
    try:
        # Если включен фильтр, сужаем датафрейм для поиска
        search_df = df
        if filter_search_by_topics and selected_topics:
            mask = df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))
            search_df = df[mask]

            # Подрезаем/назначаем эмбеддинги для search_df, чтобы они соответствовали строкам
            # Берём полный тензор из оригинального df.attrs['phrase_embs'] и индексируем его по индексам search_df
            full_embs = df.attrs.get('phrase_embs', None)
            if full_embs is not None:
                try:
                    indices = search_df.index.tolist()
                    if isinstance(full_embs, torch.Tensor):
                        if indices:
                            # индексируем тензор по оригинальным индексам (они совпадают с порядком построения)
                            search_df.attrs['phrase_embs'] = full_embs[indices]
                        else:
                            # пустой набор — создаём пустой тензор нужной ширины
                            search_df.attrs['phrase_embs'] = full_embs.new_empty((0, full_embs.size(1)))
                    else:
                        # если это numpy array или похожее
                        import numpy as np
                        arr = np.asarray(full_embs)
                        search_df.attrs['phrase_embs'] = arr[indices]
                except Exception:
                    # В крайнем случае — пересчитаем эмбеддинги для search_df (медленнее, но безопасно)
                    model = get_model()
                    if not search_df.empty:
                        search_df.attrs['phrase_embs'] = model.encode(search_df['phrase_proc'].tolist(), convert_to_tensor=True)
                    else:
                        search_df.attrs['phrase_embs'] = None

        # Проверка на пустой результат
        if search_df.empty:
            st.warning("Нет данных для поиска по выбранным тематикам.")
        else:
            results = semantic_search(query, search_df)
            exact_results = keyword_search(query, search_df)

            # Запись в лог
            log_query(
                query,
                semantic_count=len(results),
                keyword_count=len(exact_results),
                status="найдено" if results or exact_results else "не найдено"
            )

            if results:
                st.markdown("### 🔍 Результаты умного поиска:")
                for score, phrase_full, topics, comment in results:
                    with st.container():
                        st.markdown(
                            f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                                <div style="font-size: 18px; font-weight: 600; color: #333;">🧠 {phrase_full}</div>
                                <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                                <div style="margin-top: 2px; font-size: 13px; color: #999;">🎯 Релевантность: {score:.2f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий", expanded=False):
                                st.markdown(comment)
            else:
                st.warning("Совпадений не найдено в умном поиске.")

            if exact_results:
                st.markdown("### 🧷 Точный поиск:")
                for phrase, topics, comment in exact_results:
                    with st.container():
                        st.markdown(
                            f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                                <div style="font-size: 18px; font-weight: 600; color: #333;">📌 {phrase}</div>
                                <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий", expanded=False):
                                st.markdown(comment)
            else:
                st.info("Ничего не найдено в точном поиске.")

    except Exception as e:
        st.error(f"Ошибка при обработке запроса: {e}")

# Блок логов
with st.expander("⚙️ Логи (для админов)", expanded=False):
    if st.button("⬇️ Скачать логи"):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "rb") as f:
                st.download_button("Скачать как CSV", f.read(), file_name="logs.csv", mime="text/csv")
        else:
            st.info("Файл логов отсутствует")

    if st.button("🗑 Очистить логи"):
        if os.path.exists(LOG_FILE):
            open(LOG_FILE, "w").close()
        st.success("Логи очищены!")
