# utils.py

import os
import zipfile
import numpy as np
import pandas as pd
import faiss
import gdown
from sentence_transformers import SentenceTransformer
from pymorphy2 import MorphAnalyzer

# Инициализируем морфологический анализатор и кэш лемм для ускорения
morph = MorphAnalyzer()
_lemma_cache = {}

def lemmatize(text):
    """
    Лемматизация входного текста: разбиение на слова и получение нормальной формы.
    Используется pymorphy2 (MorphAnalyzer):contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}.
    """
    text = str(text).lower()
    if text in _lemma_cache:
        return _lemma_cache[text]
    words = [w for w in re.findall(r'\w+', text, flags=re.U)]
    lemmas = []
    for w in words:
        parse = morph.parse(w)
        if parse:
            lemmas.append(parse[0].normal_form)
    lemma_text = " ".join(lemmas)
    _lemma_cache[text] = lemma_text
    return lemma_text

def load_model():
    """
    Загружает zip-архив модели SentenceTransformer с Google Drive по ID, распаковывает,
    затем инициализирует и возвращает модель из локального каталога 'fine_tuned_model'.
    Использует gdown для загрузки по ID:contentReference[oaicite:10]{index=10}.
    """
    model_id = "1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf"
    zip_path = "fine_tuned_model.zip"
    model_dir = "fine_tuned_model"
    # Скачиваем, если ещё не скачано
    if not os.path.isdir(model_dir):
        url = f"https://drive.google.com/uc?id={model_id}"
        gdown.download(url, zip_path, quiet=False)
        # Распаковываем архив
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    # Загружаем модель из распакованной директории:contentReference[oaicite:11]{index=11}
    model = SentenceTransformer(model_dir)
    return model

def load_data(urls):
    """
    Загружает данные из списка URL GitHub (CSV или XLSX), объединяет их в один DataFrame.
    Обрабатывает столбцы: разбивает фразы на подфразы, приводит к строкам, очищает.
    Вычисляет леммы и эмбеддинги для фраз.
    """
    dfs = []
    for url in urls:
        try:
            if url.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(url)
            else:
                df = pd.read_csv(url)
        except Exception as e:
            # Если прямая загрузка из URL не работает, можно скачать файл и прочитать локально
            raise RuntimeError(f"Не удалось загрузить данные по адресу {url}: {e}")
        dfs.append(df)
    # Конкатенация данных из разных файлов:contentReference[oaicite:12]{index=12}
    data = pd.concat(dfs, ignore_index=True)
    # Обрабатываем колонки как строки
    for col in ['phrase', 'phrase_full', 'topics', 'comment']:
        if col in data.columns:
            data[col] = data[col].astype(str)
        else:
            data[col] = ""
    # Разбиваем phrase на подфразы (например, по запятым или точкам)
    import re
    data['subphrases'] = data['phrase'].apply(lambda x: [s.strip() for s in re.split(r'[.,;:!?]', x) if s.strip()])
    # Препроцессинг: приводим к нижнему регистру, убираем лишние символы
    data['phrase_full'] = data['phrase_full'].str.strip()
    data['topics'] = data['topics'].str.strip()
    data['comment'] = data['comment'].str.strip()
    # Лемматизация фраз и комментариев
    data['phrase_lemm'] = data['phrase'].apply(lemmatize)
    data['comment_lemm'] = data['comment'].apply(lemmatize)
    # Загружаем модель для эмбеддингов
    model = load_model()
    # Вычисляем эмбеддинги для полной фразы phrase_full
    texts = data['phrase_full'].tolist()
    if texts:
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # L2-нормализация эмбеддингов для дальнейшего использования IndexFlatIP
        faiss.normalize_L2(embeddings)  # подготовка к косинусному поиску:contentReference[oaicite:13]{index=13}
        data['embeddings'] = list(embeddings)
    else:
        data['embeddings'] = []
    return data

def build_faiss_index(embeddings):
    """
    Строит FAISS-индекс IndexFlatIP по L2-нормализованным эмбеддингам.
    """
    if len(embeddings) == 0:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (после L2-нормализации = косинусная близость)
    index.add(embeddings)
    return index

def keyword_search(query, df):
    """
    Точный поиск по леммам: ищет вхождение лемматизированного запроса
    в столбцах phrase_lemm или comment_lemm DataFrame.
    Используется pandas .str.contains():contentReference[oaicite:14]{index=14} для фильтрации.
    """
    q = lemmatize(query)
    mask_phrase = df['phrase_lemm'].str.contains(q, case=False, na=False)
    mask_comment = df['comment_lemm'].str.contains(q, case=False, na=False)
    result = df[mask_phrase | mask_comment]
    return result

def semantic_search(query, df, index, embeddings, model, threshold=0.5, top_k=5):
    """
    Семантический поиск: кодирует запрос, нормализует вектор, ищет ближайшие top_k
    в FAISS-индексе по внутреннему произведению, фильтруя по порогу threshold:contentReference[oaicite:15]{index=15}.
    Возвращает DataFrame с найденными записями.
    """
    if index is None or df.empty:
        return df.iloc[0:0]
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)  # L2-нормализуем вектор запроса:contentReference[oaicite:16]{index=16}
    # Поиск top_k
    D, I = index.search(q_emb, top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1 or score < threshold:
            continue
        hits.append(idx)
    return df.iloc[hits]

