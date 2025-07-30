import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2
import functools

# ---------- модель и морфологический разбор ----------
@functools.lru_cache(maxsize=1)
def get_model():
    import os
    import zipfile
    import gdown

    model_path = "fine_tuned_model"
    model_zip  = "fine_tuned_model.zip"
    file_id    = "1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf"

    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_zip, quiet=False)
        with zipfile.ZipFile(model_zip, 'r') as zf:
            zf.extractall(model_path)

    return SentenceTransformer(model_path)

@functools.lru_cache(maxsize=1)
def get_morph():
    return pymorphy2.MorphAnalyzer()

def preprocess(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())

def lemmatize(word):
    return get_morph().parse(word)[0].normal_form

@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word):
    return lemmatize(word)

# ---------- словарь синонимов ----------
SYNONYM_GROUPS = []
SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    lemmas = {" ".join(lemmatize_cached(w) for w in g.split()) for g in group}
    for lemma in lemmas:
        SYNONYM_DICT[lemma] = lemmas

# ---------- загрузка данных ----------
def load_file(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")

    if url.endswith(".csv"):
        df = pd.read_csv(BytesIO(resp.content))
    else:
        df = pd.read_excel(BytesIO(resp.content))

    topic_cols = [c for c in df.columns if c.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df["topics"] = df[topic_cols].astype(str).agg(lambda x: [v for v in x if v and v != "nan"], axis=1)
    df["phrase_full"] = df["phrase"]
    df["phrase_proc"] = df["phrase"].apply(preprocess)
    df["phrase_lemmas"] = df["phrase_proc"].apply(
        lambda t: {" ".join(lemmatize_cached(w) for w in re.findall(r"\w+", t))}
    )

    model = get_model()
    df.attrs["phrase_embs"] = model.encode(df["phrase_proc"].tolist(), convert_to_tensor=True)

    if "comment" not in df.columns:
        df["comment"] = ""

    return df[["phrase", "phrase_proc", "phrase_full", "phrase_lemmas", "topics", "comment"]]

def load_all_files():
    GITHUB_FILES = [
        "https://raw.githubusercontent.com/d3lskx05/data-assistanttestx/main/data6.xlsx",
        "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data21.xlsx",
        "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data31.xlsx"
    ]
    dfs = []
    for url in GITHUB_FILES:
        try:
            dfs.append(load_file(url))
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")

    df = pd.concat(dfs, ignore_index=True)

    # 🔥 Создаём индекс для точного поиска
    df.attrs["keyword_index"] = {}
    for idx, row in df.iterrows():
        for lemma in row["phrase_lemmas"]:
            df.attrs["keyword_index"].setdefault(lemma, set()).add(idx)
            if lemma in SYNONYM_DICT:
                for syn in SYNONYM_DICT[lemma]:
                    df.attrs["keyword_index"].setdefault(syn, set()).add(idx)

    return df

# ---------- поиск ----------
def semantic_search(query, df, top_k=5, threshold=0.5):
    model = get_model()
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    phrase_embs = df.attrs["phrase_embs"]

    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]
    results = [
        (float(score), df.iloc[idx]["phrase_full"], df.iloc[idx]["topics"], df.iloc[idx]["comment"])
        for idx, score in enumerate(sims) if float(score) >= threshold
    ]
    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    return results

def keyword_search(query, df):
    query_proc = preprocess(query)
    query_lemmas = {" ".join(lemmatize_cached(w) for w in re.findall(r"\w+", query_proc))}

    matched_indices = set()
    for lemma in query_lemmas:
        matched_indices |= df.attrs["keyword_index"].get(lemma, set())

    return [(df.iloc[i]["phrase_full"], df.iloc[i]["topics"], df.iloc[i]["comment"]) for i in matched_indices]

def filter_by_topics(results, selected_topics):
    if not selected_topics:
        return results

    filtered = []
    for item in results:
        if len(item) == 4:
            score, phrase, topics, comment = item
            if set(topics) & set(selected_topics):
                filtered.append((score, phrase, topics, comment))
        else:
            phrase, topics, comment = item
            if set(topics) & set(selected_topics):
                filtered.append((phrase, topics, comment))
    return filtered
