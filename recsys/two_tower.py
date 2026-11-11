"""Two-Tower: двухбашенная модель отбора кандидатов.

Так устроен этап retrieval у YouTube (Covington et al., 2016; Yi et al.,
2019), Pinterest, Spotify и других сервисов с большим каталогом.
Пользователь и фильм кодируются двумя независимыми сетями («башнями») в
векторы одной размерности, score это их скалярное произведение. Башни не
смотрят друг на друга, поэтому векторы всех фильмов считаются один раз
после обучения, а top-k для пользователя находится поиском ближайших
векторов. В проде это FAISS, HNSW или ScaNN; здесь каталог маленький,
хватает обычного умножения матриц.

Башня фильма: эмбеддинг id + жанры + десятилетие. У фильма без оценок
эмбеддинг id ничему не обучился, поэтому для него берётся только
контентная часть. Так модель может советовать и новые фильмы (холодный
старт фильма), чего не умеет KNN.

Башня пользователя: эмбеддинг id + среднее эмбеддингов последних
понравившихся фильмов. При обучении эмбеддинг id случайно обнуляется
(user_dropout), поэтому модель умеет работать и по одной истории: вектор
нового пользователя считается без переобучения (fold_in).

Обучение: sampled softmax с in-batch negatives (Yi et al., 2019). В батче
из B пар (пользователь, понравившийся фильм) фильм каждой пары служит
отрицательным примером для остальных B-1 пользователей. Популярные фильмы
попадают в негативы чаще, и модель их занижала бы; это исправляет
logQ-поправка: из логита вычитается log вероятности фильма попасть в батч.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.data import LIKE_THRESHOLD, MovieLens
from recsys.models import Recommender


class _Towers(nn.Module):
    def __init__(self, n_users: int, genres: torch.Tensor, decades: torch.Tensor, has_id: torch.Tensor, dim: int):
        super().__init__()
        n_items, n_genres = genres.shape
        self.register_buffer("genres", genres)  # фильм × жанр, 0/1
        self.register_buffer("decades", decades)  # код десятилетия
        self.register_buffer("has_id", has_id)  # 1, если у фильма есть оценки в train
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.genre_proj = nn.Linear(n_genres, dim, bias=False)
        self.decade_emb = nn.Embedding(int(decades.max()) + 1, dim)
        self.user_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.item_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        for emb in (self.user_emb, self.item_emb, self.decade_emb):
            nn.init.normal_(emb.weight, std=0.1)

    def item_input(self, items: torch.Tensor) -> torch.Tensor:
        """Вход башни фильма: id (если он чему-то обучился) + жанры + десятилетие."""
        return (
            self.item_emb(items) * self.has_id[items].unsqueeze(-1)
            + self.genre_proj(self.genres[items])
            + self.decade_emb(self.decades[items])
        )

    def item_tower(self, items: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_mlp(self.item_input(items)), dim=-1)

    def user_tower(self, users: torch.Tensor, hist: torch.Tensor, mask: torch.Tensor,
                   user_keep: torch.Tensor | None = None) -> torch.Tensor:
        """users: (B,), hist: (B, L) индексы фильмов (-1 это паддинг),
        mask: (B, L) bool, какие позиции истории учитывать,
        user_keep: (B, 1) 0/1, оставить ли эмбеддинг id пользователя."""
        h = self.item_input(hist.clamp(min=0)) * mask.unsqueeze(-1)
        mean = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        u = self.user_emb(users)
        if user_keep is not None:
            u = u * user_keep
        return F.normalize(self.user_mlp(torch.cat([u, mean], dim=-1)), dim=-1)


class TwoTower(Recommender):
    """Двухбашенная модель с in-batch negatives и logQ-поправкой."""

    name = "TwoTower"

    def __init__(self, dim: int = 64, epochs: int = 20, batch_size: int = 512, lr: float = 3e-3,
                 weight_decay: float = 1e-6, temperature: float = 0.05, user_dropout: float = 0.3,
                 max_history: int = 50, seed: int = 0, verbose: bool = False):
        self.dim, self.epochs, self.batch_size, self.lr = dim, epochs, batch_size, lr
        self.weight_decay, self.temperature, self.user_dropout = weight_decay, temperature, user_dropout
        self.max_history, self.seed, self.verbose = max_history, seed, verbose
        self.train_seconds = 0.0
        self.losses: list[float] = []

    # --- Признаки --------------------------------------------------------------

    def _item_features(self, data: MovieLens):
        movies = data.movies.set_index("movieId").loc[self.item_ids]
        genres = movies["genres"].str.get_dummies("|").drop(columns=["(no genres listed)"], errors="ignore")
        years = movies["year"].astype("float").fillna(0).to_numpy()
        _, decade_codes = np.unique((years // 10 * 10).astype(int), return_inverse=True)
        return (
            torch.tensor(genres.to_numpy(dtype=np.float32)),
            torch.tensor(decade_codes.reshape(-1), dtype=torch.long),
            torch.tensor((self.popularity > 0).astype(np.float32)),
        )

    def _history(self, liked) -> np.ndarray:
        """Последние max_history понравившихся фильмов каждого пользователя, -1 это паддинг."""
        H = np.full((len(self.user_ids), self.max_history), -1, dtype=np.int64)
        liked = liked.sort_values(["userId", "timestamp"])
        from_end = liked.groupby("userId").cumcount(ascending=False).to_numpy()
        keep = from_end < self.max_history
        u = liked["userId"].map(self.user_index).to_numpy()[keep]
        i = liked["movieId"].map(self.item_index).to_numpy()[keep]
        H[u, from_end[keep]] = i
        return H

    # --- Обучение --------------------------------------------------------------

    def _fit(self, train, data: MovieLens):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        liked = train[train["rating"] >= LIKE_THRESHOLD]

        genres, decades, has_id = self._item_features(data)
        self.model = _Towers(len(self.user_ids), genres, decades, has_id, self.dim)
        self.H = torch.tensor(self._history(liked))

        users = torch.tensor(liked["userId"].map(self.user_index).to_numpy(), dtype=torch.long)
        items = torch.tensor(liked["movieId"].map(self.item_index).to_numpy(), dtype=torch.long)
        # logQ: вероятность фильма попасть в батч пропорциональна его частоте в парах.
        counts = np.bincount(items.numpy(), minlength=len(self.item_ids)).astype(np.float64)
        log_q = torch.tensor(np.log(np.maximum(counts, 1) / len(items)), dtype=torch.float32)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = len(users)
        self.losses = []
        t0 = time.perf_counter()
        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.tensor(rng.permutation(n))
            total, batches = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                if len(idx) < 2:
                    continue
                ub, ib = users[idx], items[idx]
                hist = self.H[ub]
                # Целевой фильм в историю не входит, иначе модель его просто «подсмотрит».
                mask = (hist >= 0) & (hist != ib.unsqueeze(1))
                keep = (torch.rand(len(idx), 1) >= self.user_dropout).float()
                uvec = self.model.user_tower(ub, hist, mask, keep)
                ivec = self.model.item_tower(ib)
                logits = uvec @ ivec.T / self.temperature - log_q[ib].unsqueeze(0)
                loss = F.cross_entropy(logits, torch.arange(len(idx)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                batches += 1
            self.losses.append(total / max(batches, 1))
            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(f"  TwoTower: эпоха {epoch + 1}/{self.epochs}, loss {self.losses[-1]:.3f}", flush=True)
        self.train_seconds = time.perf_counter() - t0

        self.model.eval()
        with torch.no_grad():
            self.item_vecs = self.model.item_tower(torch.arange(len(self.item_ids))).numpy()
            all_users = torch.arange(len(self.user_ids))
            self.user_vecs = self.model.user_tower(all_users, self.H, self.H >= 0).numpy()

    # --- Использование ---------------------------------------------------------

    def scores(self, uidx):
        return self.item_vecs @ self.user_vecs[uidx]

    def fold_in(self, liked_movie_ids: list) -> np.ndarray:
        """Score для нового пользователя по списку понравившихся фильмов, без переобучения."""
        idx = [self.item_index[m] for m in liked_movie_ids if m in self.item_index][-self.max_history :]
        hist = torch.full((1, self.max_history), -1, dtype=torch.long)
        if idx:
            hist[0, : len(idx)] = torch.tensor(idx)
        with torch.no_grad():
            u = self.model.user_tower(torch.zeros(1, dtype=torch.long), hist, hist >= 0, torch.zeros(1, 1))
        return self.item_vecs @ u.numpy()[0]

    def similar_items(self, movie_id, k: int = 10) -> list:
        """Ближайшие фильмы в пространстве эмбеддингов."""
        i = self.item_index[movie_id]
        sims = self.item_vecs @ self.item_vecs[i]
        sims[i] = -np.inf
        return self.item_ids[np.argsort(-sims)[:k]].tolist()
