# utils.py

import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2
import streamlit as st
import zipfile
import os
import gdown

# ---------- модель и морфоанализ ----------

@st.cache_resource
def get_model():
    model_path = "fine_tuned_model"
    model_zip  = "fine_tuned_model.zip"
    file_id    = "1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf"

    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_zip, quiet=False)
        with zipfile.ZipFile(model_zip, 'r') as zf:
            zf.extractall(model_path)

    return SentenceTransformer(model_path)

@st.cache_resource
def get_morph():
    return pymorphy2.MorphAnalyzer()

# ---------- обработка текста ----------

def preprocess(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())

def lemmatize(word):
    return get_morph().parse(word)[0].normal_form

@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word):
    return lemmatize(word)

SYNONYM_GROUPS = []
SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    lemmas = {lemmatize(w.lower()) for w in group}
    for lemma in lemmas:
        SYNONYM_DICT[lemma] = lemmas

GITHUB_URLS = [
    "https://raw.githubusercontent.com/skatzrskx55q/data-assistant-vfiziki/main/data6.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data21.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data31.xlsx"
]

# ---------- разбивка фраз ----------

def split_by_slash(phrase: str):
    phrase = phrase.strip()
    parts  = []
    for segment in phrase.split("|"):
        segment = segment.strip()
        if "/" in segment:
            tokens = [p.strip() for p in segment.split("/") if p.strip()]
            if len(tokens) == 2:
                m = re.match(r"^(.*?\b)?(\w+)\s*/\s*(\w+)(\b.*?)?$", segment)
                if m:
                    prefix = (m.group(1) or "").strip()
                    first  = m.group(2).strip()
                    second = m.group(3).strip()
                    suffix = (m.group(4) or "").strip()
                    parts.append(" ".join(filter(None, [prefix, first,  suffix])))
                    parts.append(" ".join(filter(None, [prefix, second, suffix])))
                    continue
            parts.extend(tokens)
        else:
            parts.append(segment)
    return [p for p in parts if p]

# ---------- загрузка данных ----------

def load_table(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    data = BytesIO(resp.content)
    if url.lower().endswith(".csv"):
        df = pd.read_csv(data)
    else:
        df = pd.read_excel(data)

    topic_cols = [c for c in df.columns if c.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df["topics"] = df[topic_cols].astype(str).agg(lambda x: [v for v in x if v and v != "nan"], axis=1)
    df["phrase_full"] = df["phrase"]
    df["phrase_list"] = df["phrase"].apply(split_by_slash)
    df = df.explode("phrase_list", ignore_index=True)
    df["phrase"] = df["phrase_list"]
    df["phrase_proc"] = df["phrase"].apply(preprocess)
    df["phrase_lemmas"] = df["phrase_proc"].apply(
        lambda t: {lemmatize_cached(w) for w in re.findall(r"\w+", t)}
    )

    model = get_model()
    df.attrs["phrase_embs"] = model.encode(df["phrase_proc"].tolist(), convert_to_tensor=True)

    if "comment" not in df.columns:
        df["comment"] = ""

    return df[["phrase", "phrase_proc", "phrase_full", "phrase_lemmas", "topics", "comment"]]

def load_all_tables():
    dfs = []
    for url in GITHUB_URLS:
        try:
            dfs.append(load_table(url))
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

# ---------- поиск ----------

def deduplicate_results(results):
    best = {}
    for item in results:
        key = item[1] if len(item) == 4 else item[0]
        score = item[0] if len(item) == 4 else 1.0
        if key not in best or score > (best[key][0] if len(best[key]) == 4 else 1.0):
            best[key] = item
    return list(best.values())

def semantic_search(query, df, top_k=5, threshold=0.5):
    model = get_model()
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, df.attrs["phrase_embs"], top_k=top_k)[0]
    results = [
        (float(hit["score"]),
         df.iloc[hit["corpus_id"]]["phrase_full"],
         df.iloc[hit["corpus_id"]]["topics"],
         df.iloc[hit["corpus_id"]]["comment"])
        for hit in hits if float(hit["score"]) >= threshold
    ]
    return deduplicate_results(results)

def keyword_search(query, df):
    query_proc = preprocess(query)
    query_words = re.findall(r"\w+", query_proc)
    query_lemmas = [lemmatize_cached(w) for w in query_words]

    matched = []
    for row in df.itertuples():
        lemma_match = all(
            any(ql in SYNONYM_DICT.get(pl, {pl}) for pl in row.phrase_lemmas)
            for ql in query_lemmas
        )
        partial_match = all(q in row.phrase_proc for q in query_words)
        if lemma_match or partial_match:
            matched.append((row.phrase_full, row.topics, row.comment))
    return deduplicate_results(matched)
