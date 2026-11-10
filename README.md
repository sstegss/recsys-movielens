# Рекомендательная система фильмов на MovieLens

ПрИнж, семестр 1, задание про RecSys. Делал один.

Семь моделей рекомендаций на датасете [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) и их сравнение на отложенной по времени выборке. Разбор проблем подходов, проблем в продакшене и метрик для мониторинга лежит в [REPORT.md](REPORT.md).

| Модель | Тип | NDCG@10 | HitRate@10 |
|---|---|---|---|
| Popular | эвристика: самые смотримые | 0.074 | 0.32 |
| TopRated | эвристика: лучшие по сглаженной средней оценке | 0.064 | 0.29 |
| GenrePopular | эвристика: популярное в любимых жанрах | 0.045 | 0.24 |
| ContentBased | по жанрам, тегам и десятилетию (TF-IDF) | 0.011 | 0.07 |
| ItemKNN | коллаборативная, item-based | 0.085 | 0.39 |
| UserKNN | коллаборативная, user-based | 0.087 | 0.37 |
| Hybrid | ItemKNN + ContentBased + Popular | **0.104** | **0.44** |

## Запуск

Нужен Python 3.12. Датасет (около 1 МБ) скачается сам при первом запуске.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python evaluate.py          # сравнение моделей, около 3 минут на CPU
python demo.py --user 1     # рекомендации всех моделей для пользователя
python demo.py --movie 1    # похожие фильмы для Toy Story
```

`evaluate.py` сохраняет таблицы в [results/](results/): метрики всех моделей, подбор весов гибрида и анализ холодного старта.

## Структура

```
recsys/
  data.py        # загрузка датасета и разбиение train/test по времени
  models.py      # все модели с общим интерфейсом fit / recommend
  metrics.py     # Precision, Recall, NDCG, HitRate, Coverage, Novelty, Personalization
  evaluation.py  # прогон модели и подсчёт метрик
evaluate.py      # сравнение моделей, подбор весов гибрида, холодный старт
demo.py          # живые рекомендации
REPORT.md        # отчёт: части I–III задания
```

Данные MovieLens принадлежат GroupLens Research (University of Minnesota) и в репозиторий не входят, скрипт качает их с их сайта.
