# utils.py
import pandas as pd
import gdown
import os
from functools import lru_cache
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_data
def load_data():
    url_csv = "https://raw.githubusercontent.com/your/repository/main/data.csv"
    url_xlsx = "https://raw.githubusercontent.com/your/repository/main/data.xlsx"
    try:
        df = pd.read_csv(url_csv)
    except Exception:
        try:
            df = pd.read_excel(url_xlsx)
        except Exception:
            df = pd.DataFrame({
                "phrase": ["Привет мир", "Тестовая фраза"],
                "theme": ["Общее", "Тест"],
                "comment": ["Пример комментария", "Еще пример"]
            })
    return df

@lru_cache()
def load_model():
    url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
    output = "model.bin"
    if not os.path.exists(output):
        try:
            gdown.download(url, output, quiet=False)
        except Exception:
            return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    try:
        model = SentenceTransformer(output)
    except Exception:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model

def keyword_search(df, query):
    return df[df['phrase'].str.contains(query, case=False, na=False)]

def semantic_search(df, query, model, threshold=0.5):
    phrases = df['phrase'].tolist()
    embeddings = model.encode(phrases)
    q_emb = model.encode(query)
    scores = cosine_similarity([q_emb], embeddings)[0]
    sorted_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    df_sorted = df.iloc[sorted_indices].copy()
    df_sorted['score'] = [scores[i] for i in sorted_indices]
    df_filtered = df_sorted[df_sorted['score'] >= threshold]
    df_filtered = df_filtered.drop(columns=['score'])
    return df_filtered

def log_query(query, status, file_path="/tmp/query_log.txt"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {query} - {status}\n")
