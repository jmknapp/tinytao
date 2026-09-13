"""Cluster exact solvers by aligned behavior, not raw weights."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


def cluster_from_distance(dist: np.ndarray, n_clusters: int | None = None, threshold: float = 0.15) -> np.ndarray:
    """Agglomerative clustering on a precomputed distance matrix.

    If ``n_clusters`` is None, cut the tree at ``threshold`` (distance).
    """
    if dist.shape[0] < 2:
        return np.zeros(dist.shape[0], dtype=np.int64)
    kwargs: dict[str, Any] = {"metric": "precomputed", "linkage": "average"}
    if n_clusters is None:
        kwargs["n_clusters"] = None
        kwargs["distance_threshold"] = threshold
    else:
        kwargs["n_clusters"] = min(n_clusters, dist.shape[0])
    model = AgglomerativeClustering(**kwargs)
    return model.fit_predict(dist).astype(np.int64)


def cluster_from_features(feats: np.ndarray, n_clusters: int) -> np.ndarray:
    n = feats.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    k = min(n_clusters, n)
    # Ward on Euclidean features (already permutation-invariant).
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(feats).astype(np.int64)


def pca_embed(feats: np.ndarray, dim: int = 2) -> np.ndarray:
    if feats.shape[0] < 2:
        return np.zeros((feats.shape[0], dim))
    dim = min(dim, feats.shape[0], feats.shape[1])
    return PCA(n_components=dim).fit_transform(feats)


def cluster_report(labels: np.ndarray) -> list[dict]:
    out = []
    for k in sorted(set(labels.tolist())):
        idx = np.where(labels == k)[0]
        out.append({"cluster": int(k), "size": int(idx.size), "members": idx.tolist()})
    return out
