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
    file_id    = "1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf"  # при необходимости замените

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

SYNONYM_GROUPS = []

SYNONYM_DICT = {}
if SYNONYM_GROUPS:  # защита от пустого списка
    for group in SYNONYM_GROUPS:
        lemmas = {lemmatize(w.lower()) for w in group}
        for lemma in lemmas:
            SYNONYM_DICT[lemma] = lemmas

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/d3lskx05/data-assistanttestx/main/data4.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data21.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data31.xlsx",
]

def split_by_slash(phrase: str):
    """
    Разбивает конструкцию вида:
      'потерял симку/номер|сим карту' -> ['потерял симку', 'потерял номер', 'сим карту']
    """
    phrase = phrase.strip()
    parts  = []
    for segment in phrase.split("|"):
        segment = segment.strip()
        if "/" in segment:
            tokens = [p.strip() for p in segment.split("/") if p.strip()]
            if len(tokens) == 2:
                # Попытка восстановить общий префикс/суффикс
                m = re.match(r"^(.*?\b)?(\w+)\s*/\s*(\w+)(\b.*?)?$", segment)
                if m:
                    prefix = (m.group(1) or "").strip()
                    first  = m.group(2).strip()
                    second = m.group(3).strip()
                    suffix = (m.group(4) or "").strip()
                    parts.append(" ".join(filter(None, [prefix, first,  suffix])))
                    parts.append(" ".join(filter(None, [prefix, second, suffix])))
                    continue
            # fallback: просто берём всё, что слева/справа от /
            parts.extend(tokens)
        else:
            parts.append(segment)
    return [p for p in parts if p]

# ---------- загрузка данных ----------

def load_excel(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")

    df = pd.read_excel(BytesIO(resp.content))

    # Проверки на обязательные колонки
    if "phrase" not in df.columns:
         raise KeyError("Не найдена колонка 'phrase' в файле.")
    topic_cols = [c for c in df.columns if c.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics*")

    # Список тем
    df["topics"] = df[topic_cols].agg(lambda x: [v for v in x if pd.notna(v) and str(v).strip()], axis=1)

    # Сохраняем оригинал
    df["phrase"] = df["phrase"].astype(str)  # защита от NaN
    df["raw_phrase"] = df["phrase"]          # 👈 оригинал до разбиения

    # Разбиваем на подфразы
    df["phrase_list"] = df["phrase"].apply(split_by_slash)
    df = df.explode("phrase_list", ignore_index=True)

    # Подфраза, по которой ищем
    df["phrase_sub"] = df["phrase_list"].fillna("").astype(str).str.strip()

    # Отбрасываем пустые
    df = df[df["phrase_sub"] != ""].reset_index(drop=True)

    # Полезные поля
    df["phrase"] = df["phrase_sub"]      # совместимость с остальным кодом
    df["phrase_full"] = df["raw_phrase"] # 👈 показываем в выдаче оригинал (с / и |)

    # Обработка текста
    df["phrase_proc"] = df["phrase"].apply(preprocess)
    df["phrase_lemmas"] = df["phrase_proc"].apply(
        lambda t: {lemmatize_cached(w) for w in re.findall(r"\w+", t)}
    )

    # Эмбеддинги по подфразам
    model = get_model()
    df.attrs["phrase_embs"] = model.encode(df["phrase_proc"].tolist(), convert_to_tensor=True)

    # Комментарий
    if "comment" not in df.columns:
        df["comment"] = ""

    return df[["phrase", "phrase_proc", "phrase_full", "phrase_lemmas", "topics", "comment"]]

def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            dfs.append(load_excel(url))
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

# ---------- удаление дублей ----------

def _score_of(item):
    """Возвращает числовой score из кортежа результата."""
    return item[0] if len(item) == 4 else 1.0

def _phrase_full_of(item):
    """Возвращает phrase_full из кортежа результата."""
    return item[1] if len(item) == 4 else item[0]

def _topics_of(item):
    return item[2] if len(item) == 4 else item[1]

def _comment_of(item):
    return item[3] if len(item) == 4 else item[2]

def deduplicate_results(results, merge_topics=True, merge_comments=True):
    """
    Удаляет дубликаты по phrase_full. Если merge_topics=True,
    объединяет списки тем из разных записей одного raw_phrase.
    """
    best = {}
    for item in results:
        key   = _phrase_full_of(item)
        score = _score_of(item)
        if key not in best:
            best[key] = item
        else:
            # Выбираем лучший по score
            if score > _score_of(best[key]):
                best[key] = item
            # Обновляем темы/комментарий при желании
            if merge_topics:
                merged_topics = list({*map(str, _topics_of(best[key])), *map(str, _topics_of(item))})
                if len(best[key]) == 4:
                    best[key] = ( _score_of(best[key]), key, merged_topics, _comment_of(best[key]) )
                else:
                    best[key] = ( key, merged_topics, _comment_of(best[key]) )
            if merge_comments and _comment_of(item) and _comment_of(item) not in _comment_of(best[key]):
                if len(best[key]) == 4:
                    best[key] = ( _score_of(best[key]), key, _topics_of(best[key]),
                                  (_comment_of(best[key]) + " | " + _comment_of(item)).strip(" |") )
                else:
                    best[key] = ( key, _topics_of(best[key]),
                                  (_comment_of(best[key]) + " | " + _comment_of(item)).strip(" |") )
    return list(best.values())

# ---------- поиск ----------

def semantic_search(query, df, top_k=5, threshold=0.5):
    model       = get_model()
    query_proc  = preprocess(query)
    query_emb   = model.encode(query_proc, convert_to_tensor=True)
    phrase_embs = df.attrs["phrase_embs"]

    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]

    # Собираем кандидатов
    cand = []
    for idx, score in enumerate(sims):
        score_f = float(score)
        if score_f >= threshold:
            row = df.iloc[idx]
            cand.append((score_f, row["phrase_full"], row["topics"], row["comment"]))

    # Сортируем/ограничиваем
    cand = sorted(cand, key=lambda x: x[0], reverse=True)[:top_k * 4]  # берём чуть больше перед дедупом

    # Дедуп по исходной фразе
    return deduplicate_results(cand)

def keyword_search(query, df):
    query_proc   = preprocess(query)
    query_words  = re.findall(r"\w+", query_proc)
    query_lemmas = [lemmatize_cached(w) for w in query_words]

    matched = []
    for row in df.itertuples():
        # row.phrase_lemmas: set
        lemma_match = all(
            any(ql in SYNONYM_DICT.get(pl, {pl}) for pl in row.phrase_lemmas)
            for ql in query_lemmas
        )
        partial_match = all(q in row.phrase_proc for q in query_words)
        if lemma_match or partial_match:
            matched.append((row.phrase_full, row.topics, row.comment))

    return deduplicate_results(matched)

# ---------- фильтрация ----------

def filter_by_topics(results, selected_topics):
    if not selected_topics:
        return results
    sel = set(selected_topics)

    filtered = []
    for item in results:
        if isinstance(item, tuple) and len(item) == 4:
            score, phrase, topics, comment = item
            topics_list = topics if isinstance(topics, (list, tuple, set)) else [topics]
            if sel & set(topics_list):
                filtered.append((score, phrase, topics, comment))
        elif isinstance(item, tuple) and len(item) == 3:
            phrase, topics, comment = item
            topics_list = topics if isinstance(topics, (list, tuple, set)) else [topics]
            if sel & set(topics_list):
                filtered.append((phrase, topics, comment))
    return filtered
