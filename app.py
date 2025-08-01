import streamlit as st
import pandas as pd
import os
from utils import load_all_excels, semantic_search, keyword_search, get_model

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")
st.title("🤖 Проверка фраз")

@st.cache_data(persist="disk")
def get_data():
    df = load_all_excels()
    model = get_model()
    df.attrs['phrase_embs'] = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)
    return df

df = get_data()

all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам (независимо от поиска):", all_topics)

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

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        results = semantic_search(query, df)
        exact_results = keyword_search(query, df)

        log_query(query, len(results), len(exact_results))

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

# 👤 Админ-зона логов (малозаметные кнопки)
with st.expander("ℹ️", expanded=False):
    st.markdown(
        "<div style='font-size: 12px; color: gray;'>Скрытые функции</div>",
        unsafe_allow_html=True
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "rb") as f:
                st.download_button("📥 Скачать лог", f, file_name="query_log.csv", mime="text/csv")
    with col2:
        if st.button("🗑 Очистить лог"):
            if os.path.exists(LOG_PATH):
                os.remove(LOG_PATH)
            st.success("Лог очищен.")
