"""Загрузка MovieLens ml-latest-small и разбиение на train/test."""

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ML_DIR = DATA_DIR / "ml-latest-small"

# Оценка 4 и выше считается «понравилось». Таких примерно половина.
LIKE_THRESHOLD = 4.0


def download():
    if (ML_DIR / "ratings.csv").exists():
        return
    print(f"Скачиваю {URL}")
    DATA_DIR.mkdir(exist_ok=True)
    with urlopen(URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DATA_DIR)


@dataclass
class MovieLens:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    tags: pd.DataFrame


def load() -> MovieLens:
    download()
    ratings = pd.read_csv(ML_DIR / "ratings.csv")
    movies = pd.read_csv(ML_DIR / "movies.csv")
    tags = pd.read_csv(ML_DIR / "tags.csv")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int64")
    return MovieLens(ratings, movies, tags)


def temporal_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    """У каждого пользователя последние по времени 20% оценок уходят в test.

    Так модель учится на прошлом и угадывает будущее, как в реальном сервисе.
    Случайное разбиение подсказало бы модели оценки «из будущего».
    """
    ratings = ratings.sort_values(["userId", "timestamp"])
    pos = ratings.groupby("userId").cumcount()
    size = ratings.groupby("userId")["movieId"].transform("size")
    is_test = pos >= (size * (1 - test_frac)).round()
    return ratings[~is_test].reset_index(drop=True), ratings[is_test].reset_index(drop=True)
