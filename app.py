# app.py

import streamlit as st
from utils import load_all_data, semantic_search, keyword_search, filter_by_topics, log_query

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")
st.title("🤖 Проверка фраз")

@st.cache_data(show_spinner=True)
def get_data():
    return load_all_data()

df = get_data()
all_topics = sorted({topic for topics in df['topics'] for topic in topics})

selected_topics = st.multiselect("Фильтр по тематикам:", all_topics)

if selected_topics:
    st.markdown("### 📂 Фразы по выбранным тематикам:")
    filtered_df = df[df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))]
    for row in filtered_df.itertuples():
        with st.container():
            st.markdown(
                f"""
                <div style='border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9;'>
                    <div style='font-size: 18px; font-weight: 600;'>📝 {row.phrase_full}</div>
                    <div style='font-size: 14px;'>🔖 Тематики: <strong>{', '.join(row.topics)}</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if row.comment and str(row.comment).strip().lower() != "nan":
                with st.expander("💬 Комментарий"):
                    st.markdown(row.comment)

query = st.text_input("Введите ваш запрос:")

if query:
    try:
        semantic_results = semantic_search(query, df)
        log_query(query, "SEMANTIC", bool(semantic_results))

        if semantic_results:
            st.markdown("### 🔍 Умный поиск:")
            for score, phrase_full, topics, comment in filter_by_topics(semantic_results, selected_topics):
                with st.container():
                    st.markdown(
                        f"""
                        <div style='border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9;'>
                            <div style='font-size: 18px; font-weight: 600;'>🧠 {phrase_full}</div>
                            <div style='font-size: 14px;'>🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                            <div style='font-size: 13px;'>🎯 Релевантность: {score:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if comment and str(comment).strip().lower() != "nan":
                        with st.expander("💬 Комментарий"):
                            st.markdown(comment)
        else:
            st.warning("Совпадений не найдено в умном поиске.")

        keyword_results = keyword_search(query, df)
        log_query(query, "KEYWORD", bool(keyword_results))

        if keyword_results:
            st.markdown("### 🧷 Точный поиск:")
            for phrase_full, topics, comment in filter_by_topics(keyword_results, selected_topics):
                with st.container():
                    st.markdown(
                        f"""
                        <div style='border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9;'>
                            <div style='font-size: 18px; font-weight: 600;'>📌 {phrase_full}</div>
                            <div style='font-size: 14px;'>🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if comment and str(comment).strip().lower() != "nan":
                        with st.expander("💬 Комментарий"):
                            st.markdown(comment)
        else:
            st.info("Ничего не найдено в точном поиске.")

    except Exception as e:
        st.error(f"Ошибка при обработке запроса: {e}")
