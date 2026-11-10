"""Метрики качества top-N рекомендаций."""

import numpy as np


def precision_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / len(relevant)


def ndcg_at_k(recs, relevant, k):
    """Учитывает позицию: попадание на 1-е место ценнее, чем на 10-е."""
    dcg = sum(1 / np.log2(i + 2) for i, m in enumerate(recs[:k]) if m in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


def hit_rate_at_k(recs, relevant, k):
    """Есть ли в выдаче хоть один фильм, который пользователю понравился."""
    return float(bool(set(recs[:k]) & relevant))


def coverage(all_recs, n_items):
    """Какая доля каталога хоть раз попала в чьи-то рекомендации."""
    return len({m for recs in all_recs for m in recs}) / n_items


def novelty(all_recs, item_popularity, n_users):
    """Средняя «неожиданность» рекомендованных фильмов, -log2 доли зрителей.

    Чем выше, тем менее заезженные фильмы советует модель.
    """
    vals = [
        -np.log2(item_popularity.get(m, 1) / n_users) for recs in all_recs for m in recs
    ]
    return float(np.mean(vals))


def personalization(all_recs):
    """1 минус среднее пересечение выдач двух случайных пользователей.

    0 значит, что всем показывают одно и то же.
    """
    rng = np.random.default_rng(0)
    sets = [set(r) for r in all_recs]
    pairs = rng.integers(0, len(sets), size=(2000, 2))
    overlaps = [
        len(sets[a] & sets[b]) / max(len(sets[a]), 1) for a, b in pairs if a != b
    ]
    return 1 - float(np.mean(overlaps))
