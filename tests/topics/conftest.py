import numpy as np
import pytest

from arxiv_atlas.embeddings.model import DIM


@pytest.fixture
def blobs():
    """Unit vectors in three tight, well separated groups of 60, shuffled."""

    def make(n_per_group=60, dim=DIM, seed=0):
        rng = np.random.default_rng(seed)
        centres = rng.normal(size=(3, dim))
        groups = np.repeat(np.arange(3), n_per_group)
        vectors = centres[groups] + 0.05 * rng.normal(size=(groups.size, dim))
        order = rng.permutation(groups.size)
        vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)
        return vectors[order], groups[order]

    return make
