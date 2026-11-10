"""Рекомендательные модели.

У всех моделей одинаковый интерфейс: fit(train, data) и recommend(user_id, k).
Внутри каждая считает score для каждого фильма каталога, дальше общий код
выкидывает уже просмотренное и берёт top-k.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.data import LIKE_THRESHOLD, MovieLens


class Recommender:
    name = "base"

    def fit(self, train: pd.DataFrame, data: MovieLens):
        # Каталог: все фильмы из movies.csv, даже те, что никто не оценил в train.
        self.item_ids = data.movies["movieId"].to_numpy()
        self.item_index = {m: i for i, m in enumerate(self.item_ids)}
        self.user_ids = np.sort(train["userId"].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}

        rows = train["userId"].map(self.user_index).to_numpy()
        cols = train["movieId"].map(self.item_index).to_numpy()
        shape = (len(self.user_ids), len(self.item_ids))
        # Матрица оценок пользователь × фильм. Пропуск значит «не смотрел».
        self.R = sp.csr_matrix((train["rating"].to_numpy(), (rows, cols)), shape=shape)
        self.popularity = np.asarray((self.R > 0).sum(axis=0)).ravel()
        self._fit(train, data)
        return self

    def _fit(self, train, data):
        pass

    def scores(self, uidx: int) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_id, k: int = 10, exclude_seen: bool = True) -> list:
        uidx = self.user_index.get(user_id)
        if uidx is None:
            # Новый пользователь: истории нет, отдаём популярное (холодный старт).
            scores = self.popularity.astype(float)
        else:
            scores = self.scores(uidx).astype(float)
            # При равных score (например, все нули, если модели не хватило
            # истории) выше идёт более популярный фильм, а не случайный.
            scores += 1e-9 * self.popularity / self.popularity.max()
            if exclude_seen:
                scores[self.R[uidx].indices] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return self.item_ids[top].tolist()


# --- Эвристики без обучения -------------------------------------------------


class Popular(Recommender):
    """Самые смотримые фильмы. Одинаковая выдача для всех."""

    name = "Popular"

    def scores(self, uidx):
        return self.popularity


class TopRated(Recommender):
    """Лучшие по средней оценке. Средняя сглажена к общей (байесовская),
    иначе наверх вылезут фильмы с одной оценкой 5.0."""

    name = "TopRated"

    def __init__(self, m: int = 20):
        self.m = m

    def _fit(self, train, data):
        sums = np.asarray(self.R.sum(axis=0)).ravel()
        global_mean = train["rating"].mean()
        self.bayes = (sums + self.m * global_mean) / (self.popularity + self.m)

    def scores(self, uidx):
        return self.bayes


class GenrePopular(Recommender):
    """Популярное в любимых жанрах пользователя.

    Любимые жанры считаются по фильмам, которым он поставил 4 и выше.
    """

    name = "GenrePopular"

    def _fit(self, train, data):
        genres = data.movies["genres"].str.get_dummies("|")
        genres = genres.drop(columns=["(no genres listed)", "IMAX"], errors="ignore")
        self.genre_names = genres.columns.tolist()
        self.G = sp.csr_matrix(genres.to_numpy(dtype=float))  # фильм × жанр
        liked = self.R.multiply(self.R >= LIKE_THRESHOLD)
        liked.data[:] = 1
        # Доля понравившихся фильмов в каждом жанре.
        prefs = liked @ self.G
        self.prefs = normalize(sp.csr_matrix(prefs), norm="l1")
        self.pop_score = np.log1p(self.popularity)

    def scores(self, uidx):
        genre_match = (self.G @ self.prefs[uidx].T).toarray().ravel()
        return genre_match * self.pop_score


# --- Content-based ----------------------------------------------------------


class ContentBased(Recommender):
    """Похожие по описанию: жанры, теги пользователей и десятилетие выхода.

    Каждый фильм — TF-IDF вектор из этих «слов». Профиль пользователя —
    сумма векторов его фильмов с весом (оценка − его средняя оценка):
    понравившееся тянет профиль к себе, не понравившееся отталкивает.
    """

    name = "ContentBased"

    def _fit(self, train, data):
        movies = data.movies.set_index("movieId").loc[self.item_ids]
        genre_tokens = movies["genres"].str.replace("(no genres listed)", "", regex=False)
        genre_tokens = genre_tokens.str.split("|").apply(lambda g: " ".join(f"g_{x}" for x in g if x))
        decade = movies["year"].apply(lambda y: f"decade_{y // 10 * 10}" if pd.notna(y) else "")

        # Теги из train-части по времени, чтобы не подсматривать в будущее.
        tags = data.tags[data.tags["timestamp"] <= train["timestamp"].max()]
        tag_tokens = (
            tags.assign(tag=tags["tag"].str.lower().str.replace(r"[^\w]+", "_", regex=True))
            .groupby("movieId")["tag"]
            .apply(" ".join)
        )
        tag_tokens = tag_tokens.reindex(movies.index).fillna("")

        docs = genre_tokens + " " + decade + " " + tag_tokens
        tfidf = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True)
        self.X = tfidf.fit_transform(docs)  # фильм × признак, строки нормированы

        R = self.R.tocsr(copy=True)
        means = np.asarray(R.sum(axis=1)).ravel() / np.diff(R.indptr)
        R.data -= np.repeat(means, np.diff(R.indptr))
        self.profiles = normalize(R @ self.X)

    def scores(self, uidx):
        return (self.X @ self.profiles[uidx].T).toarray().ravel()

    def similar_items(self, movie_id, k=10):
        """«Вам может понравиться» на странице фильма."""
        i = self.item_index[movie_id]
        sims = (self.X @ self.X[i].T).toarray().ravel()
        sims[i] = -np.inf
        return self.item_ids[np.argsort(-sims)[:k]].tolist()


# --- Коллаборативная фильтрация --------------------------------------------


def _top_k_per_row(S: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Оставить в каждой строке матрицы сходства только k самых больших."""
    S = S.tocsr()
    rows, cols, vals = [], [], []
    for r in range(S.shape[0]):
        start, end = S.indptr[r], S.indptr[r + 1]
        if end == start:
            continue
        v, c = S.data[start:end], S.indices[start:end]
        if len(v) > k:
            keep = np.argpartition(-v, k)[:k]
            v, c = v[keep], c[keep]
        rows.extend([r] * len(v))
        cols.extend(c)
        vals.extend(v)
    return sp.csr_matrix((vals, (rows, cols)), shape=S.shape)


class ItemKNN(Recommender):
    """Item-based: «с этим фильмом часто смотрят».

    Два фильма похожи, если их смотрели одни и те же люди (косинус между
    столбцами матрицы просмотров). Score фильма для пользователя — сумма
    сходств с фильмами из его истории.
    """

    name = "ItemKNN"

    def __init__(self, k: int = 50):
        self.k = k

    def _fit(self, train, data):
        B = (self.R > 0).astype(float)
        Bn = normalize(B, axis=0)
        S = (Bn.T @ Bn).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        self.S = _top_k_per_row(S, self.k)  # фильм × фильм
        self.B = B.tocsr()

    def scores(self, uidx):
        return (self.B[uidx] @ self.S.T).toarray().ravel()

    def similar_items(self, movie_id, k=10):
        """«С этим фильмом также смотрят» на странице фильма."""
        row = self.S[self.item_index[movie_id]]
        order = np.argsort(-row.data)[:k]
        return self.item_ids[row.indices[order]].tolist()


class UserKNN(Recommender):
    """User-based: «пользователи с похожим вкусом смотрели».

    Сходство пользователей — косинус между их центрированными оценками,
    так учитывается не только что смотрел, но и понравилось ли.
    Берём k самых похожих соседей и складываем их просмотры с весом сходства.
    """

    name = "UserKNN"

    def __init__(self, k: int = 50):
        self.k = k

    def _fit(self, train, data):
        R = self.R.tocsr(copy=True)
        means = np.asarray(R.sum(axis=1)).ravel() / np.diff(R.indptr)
        R.data -= np.repeat(means, np.diff(R.indptr))
        Rn = normalize(R)
        S = (Rn @ Rn.T).tocsr()
        S.setdiag(0)
        S.data[S.data < 0] = 0
        S.eliminate_zeros()
        self.S = _top_k_per_row(S, self.k)  # пользователь × пользователь
        self.B = (self.R > 0).astype(float).tocsr()

    def scores(self, uidx):
        return (self.S[uidx] @ self.B).toarray().ravel()


# --- Гибрид -----------------------------------------------------------------


def _minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class Hybrid(Recommender):
    """Взвешенная сумма нормированных score нескольких моделей.

    CF хорошо угадывает, но не знает фильмов без оценок. Content-based
    видит любой фильм с жанрами, но угадывает хуже. Смесь берёт лучшее
    от обоих, а вес популярности не даёт уйти в совсем редкие фильмы.
    """

    name = "Hybrid"

    def __init__(self, parts: dict[Recommender, float]):
        self.parts = parts

    def _fit(self, train, data):
        for model in self.parts:
            model.fit(train, data)

    def scores(self, uidx):
        return sum(w * _minmax(m.scores(uidx).astype(float)) for m, w in self.parts.items())
