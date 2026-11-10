"""Показать рекомендации разных моделей для пользователя и похожие фильмы.

python demo.py --user 1
python demo.py --movie 1   # Toy Story
"""

import argparse

from recsys import models as M
from recsys.data import LIKE_THRESHOLD, load

# Веса из evaluate.py (подобраны на валидации).
HYBRID_WEIGHTS = (0.4, 0.4, 0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=1)
    parser.add_argument("--movie", type=int, default=None)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    data = load()
    titles = data.movies.set_index("movieId")["title"]
    train = data.ratings  # для демо учимся на всех оценках

    if args.movie is not None:
        content = M.ContentBased().fit(train, data)
        item = M.ItemKNN().fit(train, data)
        print(f"Фильм: {titles[args.movie]}\n")
        print("Вам может понравиться (похожие по жанрам и тегам):")
        for m in content.similar_items(args.movie, args.k):
            print("  ", titles[m])
        print("\nС этим фильмом также смотрят (item-based CF):")
        for m in item.similar_items(args.movie, args.k):
            print("  ", titles[m])
        return

    history = train[train["userId"] == args.user].sort_values("timestamp")
    liked = history[history["rating"] >= LIKE_THRESHOLD]
    print(f"Пользователь {args.user}: {len(history)} оценок, понравилось {len(liked)}. Последние понравившиеся:")
    for m in liked["movieId"].tail(5):
        print("  ", titles[m])
    print("Недавно смотрел:")
    for m in history["movieId"].tail(5)[::-1]:
        print("  ", titles[m])

    w = HYBRID_WEIGHTS
    models = [
        M.Popular(),
        M.GenrePopular(),
        M.ContentBased(),
        M.ItemKNN(),
        M.UserKNN(),
        M.Hybrid({M.ItemKNN(): w[0], M.ContentBased(): w[1], M.Popular(): w[2]}),
    ]
    for model in models:
        model.fit(train, data)
        print(f"\n{model.name}:")
        for m in model.recommend(args.user, args.k):
            print("  ", titles[m])


if __name__ == "__main__":
    main()
