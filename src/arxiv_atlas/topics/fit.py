"""Find topics in paper vectors: UMAP to a few dimensions, then HDBSCAN.

Density-based clustering breaks down in 1024 dimensions, where all distances look
alike. UMAP first squeezes the vectors to 5 dimensions while keeping papers that were
neighbours close together; HDBSCAN then finds the dense regions. Papers in no dense
region are labelled -1 (no topic) instead of being forced into the nearest one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from arxiv_atlas.embeddings.model import Vectors

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FitParams:
    n_neighbors: int = 15  # how local the structure UMAP keeps is
    cluster_dims: int = 5
    min_cluster_size: int = 40  # the smallest group of papers that counts as a topic
    min_samples: int = 5  # higher: more papers left without a topic, cleaner topics
    seed: int = 42


@dataclass(frozen=True, slots=True)
class FitResult:
    labels: NDArray[np.int_]  # topic per paper, -1 = none; topic 0 is the largest
    coords: NDArray[np.float32]  # (n, 2) positions for the map


def fit_topics(vectors: Vectors, params: FitParams) -> FitResult:
    # Imported here: UMAP compiles code on import, which every other command would pay for.
    from sklearn.cluster import HDBSCAN
    from umap.umap_ import nearest_neighbors

    started = time.monotonic()
    # Both UMAP runs need the same neighbour graph, and finding it is the slow part.
    knn = nearest_neighbors(
        vectors,
        n_neighbors=params.n_neighbors,
        metric="cosine",
        metric_kwds=None,
        angular=False,
        random_state=np.random.RandomState(params.seed),
    )
    log.info("neighbour graph done after %.0fs", time.monotonic() - started)

    reduced = _umap(vectors, knn, params, dims=params.cluster_dims, min_dist=0.0)
    log.info("UMAP to %dD done after %.0fs", params.cluster_dims, time.monotonic() - started)
    raw = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        copy=False,  # `reduced` is not used afterwards
    ).fit_predict(reduced)
    log.info("HDBSCAN done after %.0fs", time.monotonic() - started)
    # For the map, a little spacing (min_dist) keeps points from piling on each other.
    coords = _umap(vectors, knn, params, dims=2, min_dist=0.1)
    log.info("UMAP to 2D done after %.0fs", time.monotonic() - started)
    return FitResult(labels=_by_size(raw), coords=coords)


def _umap(
    vectors: Vectors, knn: tuple[object, ...], params: FitParams, dims: int, min_dist: float
) -> NDArray[np.float32]:
    from umap.umap_ import UMAP

    reducer = UMAP(
        n_components=dims,
        n_neighbors=params.n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=params.seed,
        n_jobs=1,  # a fixed seed already forces one thread; saying so silences a warning
        precomputed_knn=knn,
        # The default layout algorithm is about to change; pin it so results stay comparable.
        compatibility_layout=True,
    )
    return np.asarray(reducer.fit_transform(vectors), dtype=np.float32)


def _by_size(labels: NDArray[np.int_]) -> NDArray[np.int_]:
    """Renumber topics so that 0 is the largest; -1 stays -1."""
    topics, sizes = np.unique(labels[labels >= 0], return_counts=True)
    new_id = {int(t): i for i, t in enumerate(topics[np.argsort(-sizes, kind="stable")])}
    return np.array([new_id.get(int(label), -1) for label in labels], dtype=np.int_)
