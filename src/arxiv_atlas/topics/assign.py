"""Put papers outside the fitting sample into topics, by their nearest fitted papers.

Each paper takes the topic that wins a vote among its k most similar sample papers,
each vote weighted by similarity, and sits on the map at the average position of the
neighbours that voted for the winner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from arxiv_atlas.embeddings.model import Vectors


@dataclass(frozen=True, slots=True)
class Reference:
    """The fitted sample: unit-length vectors with their topics and map positions."""

    vectors: Vectors
    labels: NDArray[np.int_]
    coords: NDArray[np.float32]


def assign_topics(
    reference: Reference, queries: Vectors, k: int = 10, chunk_size: int = 512
) -> tuple[NDArray[np.int_], NDArray[np.float32]]:
    k = min(k, len(reference.vectors))
    labels = np.empty(len(queries), dtype=np.int_)
    coords = np.empty((len(queries), 2), dtype=np.float32)
    n_labels = int(reference.labels.max(initial=-1)) + 2  # topics 0..n-1, plus -1
    for start in range(0, len(queries), chunk_size):
        chunk = slice(start, start + chunk_size)
        # Unit-length vectors: the dot product is the cosine similarity.
        similarity = queries[chunk] @ reference.vectors.T
        nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        weights = np.take_along_axis(similarity, nearest, axis=1)
        neighbour_labels = reference.labels[nearest]

        votes = np.zeros((len(nearest), n_labels))
        rows = np.arange(len(nearest))[:, None]
        np.add.at(votes, (rows, neighbour_labels + 1), weights)  # column 0 is -1
        winner = votes.argmax(axis=1) - 1

        voted = (neighbour_labels == winner[:, None])[..., None]
        coords[chunk] = (reference.coords[nearest] * voted).sum(axis=1) / voted.sum(axis=1)
        labels[chunk] = winner
    return labels, coords
