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

# 📌 Независимая фильтрация по темам (не влияет на поиск)
if selected_topics:
    st.markdown("### 📂 Фразы по выбранным тематикам:")
    filtered_df = df[df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))]
    for idx, row in enumerate(filtered_df.itertuples()):
        st.markdown(
            f"""
            <div style="background-color:#ffffff; border:1px solid #e0e0e0; padding:16px; border-radius:12px; margin-bottom:10px; box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:1.1em; font-weight:600; margin-bottom:6px;">{row.phrase_full}</div>
                <div style="color:#888; font-size:0.85em;">Темы: {', '.join(row.topics)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if row.comment and str(row.comment).strip().lower() != "nan":
            with st.expander("💬 Показать комментарий", expanded=False):
                st.markdown(
                    f"""
                    <div style="background-color:#f0f2f6; padding:8px 12px; border-radius:10px; margin:0 0 10px 0; font-size:0.9em; color:#333;">
                        {row.comment}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# 📥 Поисковый запрос
query = st.text_input("Введите ваш запрос:")

if query:
    try:
        results = semantic_search(query, df)
        if results:
            st.markdown("### 🔍 Результаты умного поиска:")
            for idx, (score, phrase_full, topics, comment) in enumerate(results):
                st.markdown(
                    f"""
                    <div style="background-color:#ffffff; border:1px solid #e0e0e0; padding:16px; border-radius:12px; margin-bottom:10px; box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                        <div style="font-size:1.1em; font-weight:600; margin-bottom:6px;">{phrase_full}</div>
                        <div style="color:#888; font-size:0.85em;">Темы: {', '.join(topics)}</div>
                        <div style="color:#bbb; font-size:0.75em; margin-top:4px;">Score: {score:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if comment and str(comment).strip().lower() != "nan":
                    with st.expander("💬 Показать комментарий", expanded=False):
                        st.markdown(
                            f"""
                            <div style="background-color:#f0f2f6; padding:8px 12px; border-radius:10px; margin:0 0 10px 0; font-size:0.9em; color:#333;">
                                {comment}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.warning("Совпадений не найдено в умном поиске.")

        exact_results = keyword_search(query, df)
        if exact_results:
            st.markdown("### 🧷 Точный поиск:")
            for idx, (phrase, topics, comment) in enumerate(exact_results):
                st.markdown(
                    f"""
                    <div style="background-color:#ffffff; border:1px solid #e0e0e0; padding:16px; border-radius:12px; margin-bottom:10px; box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                        <div style="font-size:1.1em; font-weight:600; margin-bottom:6px;">{phrase}</div>
                        <div style="color:#888; font-size:0.85em;">Темы: {', '.join(topics)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if comment and str(comment).strip().lower() != "nan":
                    with st.expander("💬 Показать комментарий", expanded=False):
                        st.markdown(
                            f"""
                            <div style="background-color:#f0f2f6; padding:8px 12px; border-radius:10px; margin:0 0 10px 0; font-size:0.9em; color:#333;">
                                {comment}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.info("Ничего не найдено в точном поиске.")

    except Exception as e:
        st.error(f"Ошибка при обработке запроса: {e}")
