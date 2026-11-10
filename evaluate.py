"""Сравнение всех моделей и анализ холодного старта.

Результаты печатаются и сохраняются в results/.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from recsys import models as M
from recsys.data import LIKE_THRESHOLD, load, temporal_split
from recsys.evaluation import evaluate

RESULTS = Path(__file__).parent / "results"
K = 10


def hybrid(w_item, w_content, w_pop):
    return M.Hybrid({M.ItemKNN(): w_item, M.ContentBased(): w_content, M.Popular(): w_pop})


def tune_hybrid(train, data):
    """Веса гибрида подбираются на валидации: последние 20% train каждого
    пользователя. Test в подборе не участвует, иначе оценка будет завышена."""
    sub_train, val = temporal_split(train)
    grid = [w for w in itertools.product([0.2, 0.4, 0.6, 0.8], [0, 0.2, 0.4, 0.6], [0, 0.2, 0.4]) if w[0] > 0]
    rows = []
    for w in grid:
        r = evaluate(hybrid(*w), sub_train, val, data, K)
        rows.append({"item": w[0], "content": w[1], "pop": w[2], "NDCG@10 (val)": r["NDCG@10"]})
    table = pd.DataFrame(rows).sort_values("NDCG@10 (val)", ascending=False)
    best = table.iloc[0]
    return (best["item"], best["content"], best["pop"]), table


def cold_start(train, test, data, models_fn, n_users=100, seed=0):
    """Делаем вид, что 100 случайных пользователей только пришли и успели
    оценить первые N фильмов. Всё, что они оценили позже (остаток train и test),
    становится их «будущим», которое модель должна угадать.
    N=0 значит совсем новый пользователь."""
    rng = np.random.default_rng(seed)
    cold = set(rng.choice(train["userId"].unique(), n_users, replace=False))
    is_cold = train["userId"].isin(cold)
    rows = []
    for n in [0, 1, 3, 5, 10, 20]:
        pos = train[is_cold].groupby("userId").cumcount()
        known = train[is_cold][pos < n]
        future = pd.concat([train[is_cold][pos >= n], test[test["userId"].isin(cold)]])
        tr = pd.concat([train[~is_cold], known])
        for model in models_fn():
            r = evaluate(model, tr, future, data, K, users=cold)
            rows.append({"оценок в истории": n, "model": model.name, "NDCG@10": r["NDCG@10"]})
    return pd.DataFrame(rows).pivot(index="model", columns="оценок в истории", values="NDCG@10")


def cold_items(train, test):
    """Сколько понравившихся в test фильмов вообще не встречались в train.
    Их не может посоветовать ни одна CF-модель."""
    liked = test[test["rating"] >= LIKE_THRESHOLD]
    unseen = ~liked["movieId"].isin(train["movieId"])
    return unseen.mean()


def main():
    RESULTS.mkdir(exist_ok=True)
    data = load()
    train, test = temporal_split(data.ratings)
    print(f"train {len(train)} оценок, test {len(test)} оценок, {data.ratings.userId.nunique()} пользователей, {len(data.movies)} фильмов")

    weights, grid = tune_hybrid(train, data)
    print("\nЛучшие веса гибрида на валидации (ItemKNN, Content, Popular):", weights)
    grid.head(10).to_markdown(RESULTS / "hybrid_tuning.md", index=False, floatfmt=".4f")

    models = [M.Popular(), M.TopRated(), M.GenrePopular(), M.ContentBased(), M.ItemKNN(), M.UserKNN(), hybrid(*weights)]
    table = pd.DataFrame([evaluate(m, train, test, data, K) for m in models])
    print("\n" + table.to_string(index=False, float_format="%.4f"))
    table.to_markdown(RESULTS / "metrics.md", index=False, floatfmt=".4f")

    cs = cold_start(train, test, data, lambda: [M.Popular(), M.GenrePopular(), M.ContentBased(), M.ItemKNN(), M.UserKNN(), hybrid(*weights)])
    print("\nХолодный старт, NDCG@10 в зависимости от длины истории:\n" + cs.to_string(float_format="%.4f"))
    cs.to_markdown(RESULTS / "cold_start.md", floatfmt=".4f")

    share = cold_items(train, test)
    print(f"\nПонравившихся фильмов из test, которых нет в train: {share:.1%}")
    (RESULTS / "cold_items.txt").write_text(f"{share:.4f}\n")


if __name__ == "__main__":
    main()
