# app.py

import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("🤖 Semantic Assistant")

@st.cache_data
def get_data():
    df = load_all_excels()
    from utils import get_model
    model = get_model()
    df.attrs['phrase_embs'] = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)
    return df

df = get_data()

# 🔘 Все уникальные тематики
all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам (независимо от поиска):", all_topics)

# 📌 Независимая фильтрация по темам
if selected_topics:
    st.markdown("### 📂 Фразы по выбранным тематикам:")
    filtered_df = df[df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))]
    for row in filtered_df.itertuples():
        st.markdown(f"- **{row.phrase_full}** → {', '.join(row.topics)}")
        if row.comment and str(row.comment).strip().lower() != "nan":
            with st.expander("💬 Комментарий", expanded=False):
                st.markdown(row.comment)

# 📥 Поисковый запрос
query = st.text_input("Введите ваш запрос:")

if query:
    try:
        results = semantic_search(query, df)
        if results:
            st.markdown("### 🔍 Результаты умного поиска:")
            for score, phrase_full, topics, comment in results:
                st.markdown(f"**{phrase_full}** → {', '.join(topics)} (_{score:.2f}_)")
                if comment and str(comment).strip().lower() != "nan":
                    with st.expander("💬 Комментарий", expanded=False):
                        st.markdown(comment)
                st.markdown("<hr style='margin:4px 0' />", unsafe_allow_html=True)
        else:
            st.warning("Совпадений не найдено в умном поиске.")

        exact_results = keyword_search(query, df)
        if exact_results:
            st.markdown("### 🧷 Точный поиск:")
            for phrase, topics, comment in exact_results:
                st.markdown(f"**{phrase}** → {', '.join(topics)}")
                if comment and str(comment).strip().lower() != "nan":
                    with st.expander("💬 Комментарий", expanded=False):
                        st.markdown(comment)
                st.markdown("<hr style='margin:4px 0' />", unsafe_allow_html=True)
        else:
            st.info("Ничего не найдено в точном поиске.")

    except Exception as e:
        st.error(f"Ошибка при обработке запроса: {e}")
