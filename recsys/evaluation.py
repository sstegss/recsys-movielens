"""Офлайн-оценка моделей на отложенной по времени выборке."""

import time

import numpy as np
import pandas as pd

from recsys import metrics
from recsys.data import LIKE_THRESHOLD


def relevant_items(test: pd.DataFrame) -> dict:
    """Фильмы из test, которые пользователю понравились (оценка 4+)."""
    liked = test[test["rating"] >= LIKE_THRESHOLD]
    return liked.groupby("userId")["movieId"].apply(set).to_dict()


def evaluate(model, train, test, data, k: int = 10, users=None) -> dict:
    relevant = relevant_items(test)
    if users is not None:
        relevant = {u: r for u, r in relevant.items() if u in users}

    t0 = time.perf_counter()
    model.fit(train, data)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    all_recs = {u: model.recommend(u, k) for u in relevant}
    rec_time = (time.perf_counter() - t0) / len(relevant)

    rows = [
        (
            metrics.precision_at_k(all_recs[u], rel, k),
            metrics.recall_at_k(all_recs[u], rel, k),
            metrics.ndcg_at_k(all_recs[u], rel, k),
            metrics.hit_rate_at_k(all_recs[u], rel, k),
        )
        for u, rel in relevant.items()
    ]
    p, r, n, h = np.mean(rows, axis=0)
    pop = train.groupby("movieId").size().to_dict()
    recs = list(all_recs.values())
    return {
        "model": model.name,
        f"Precision@{k}": p,
        f"Recall@{k}": r,
        f"NDCG@{k}": n,
        f"HitRate@{k}": h,
        "Coverage": metrics.coverage(recs, len(data.movies)),
        "Novelty": metrics.novelty(recs, pop, train["userId"].nunique()),
        "Personalization": metrics.personalization(recs),
        "fit, с": fit_time,
        "ответ, мс": rec_time * 1000,
    }
