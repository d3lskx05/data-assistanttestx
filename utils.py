import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2
import functools
import os
import zipfile
import gdown

# ---------- модель и морфологический разбор ----------

@functools.lru_cache(maxsize=1)
def get_model():
    model_path = "fine_tuned_model"
    model_zip = "fine_tuned_model.zip"
    file_id = "1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf"  # при необходимости замените

    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_zip, quiet=False)
        with zipfile.ZipFile(model_zip, 'r') as zf:
            zf.extractall(model_path)

    return SentenceTransformer(model_path)

@functools.lru_cache(maxsize=1)
def get_morph():
    return pymorphy2.MorphAnalyzer()

# ---------- служебные функции ----------

def preprocess(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())

def lemmatize(word):
    return get_morph().parse(word)[0].normal_form

@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word):
    return lemmatize(word)

# ---------- загрузка синонимов ----------
def load_synonyms(file_path="synonyms.csv"):
    df = pd.read_csv(file_path)
    synonym_groups = []
    for row in df["group"]:
        words = [w.strip() for w in str(row).split(",") if w.strip()]
        synonym_groups.append(set(words))
    return synonym_groups

SYNONYM_GROUPS = load_synonyms()

SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    lemmas = {lemmatize(w.lower()) for w in group}
    for lemma in lemmas:
        SYNONYM_DICT[lemma] = lemmas

def expand_query_with_synonyms(query: str):
    query_lemmas = {lemmatize_cached(w) for w in re.findall(r"\w+", query.lower())}
    expanded = set([query.lower()])
    for lemma in query_lemmas:
        if lemma in SYNONYM_DICT:
            expanded |= SYNONYM_DICT[lemma]
    return expanded

# ---------- загрузка данных ----------

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrskx55q/data-assistant-vfiziki/main/data6.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data21.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data31.xlsx"
]

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
                    parts.append(" ".join(filter(None, [prefix, first, suffix])))
                    parts.append(" ".join(filter(None, [prefix, second, suffix])))
                    continue
            parts.extend(tokens)
        else:
            parts.append(segment)
    return [p for p in parts if p]

def load_excel_or_csv(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")

    content_type = resp.headers.get("Content-Type", "")
    if "csv" in url or "text/csv" in content_type:
        df = pd.read_csv(BytesIO(resp.content))
    else:
        df = pd.read_excel(BytesIO(resp.content))

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

def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            dfs.append(load_excel_or_csv(url))
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

# ---------- удаление дублей ----------
def _score_of(item):
    return item[0] if len(item) == 4 else 1.0

def _phrase_full_of(item):
    return item[1] if len(item) == 4 else item[0]

def deduplicate_results(results):
    best = {}
    for item in results:
        key   = _phrase_full_of(item)
        score = _score_of(item)
        if key not in best or score > _score_of(best[key]):
            best[key] = item
    return list(best.values())

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
    return deduplicate_results(results)

def keyword_search(query, df):
    query_variants = expand_query_with_synonyms(query)
    matched = []

    for row in df.itertuples():
        for variant in query_variants:
            if variant in row.phrase_proc:
                matched.append((row.phrase_full, row.topics, row.comment))
                break
    return deduplicate_results(matched)

# ---------- фильтрация ----------
def filter_by_topics(df, selected_topics):
    if not selected_topics:
        return df
    return df[df["topics"].apply(lambda topics: any(t for t in topics if any(st.lower() in t.lower() for st in selected_topics)))]
