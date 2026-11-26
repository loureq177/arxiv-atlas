import numpy as np
from sklearn.metrics import adjusted_rand_score

from arxiv_atlas.topics.fit import FitParams, fit_topics

PARAMS = FitParams(n_neighbors=10, min_cluster_size=20, min_samples=5)


def test_finds_the_groups(blobs):
    vectors, groups = blobs()
    result = fit_topics(vectors, PARAMS)
    assert adjusted_rand_score(groups, result.labels) > 0.95
    assert result.coords.shape == (len(vectors), 2)


def test_topic_zero_is_the_largest(blobs):
    vectors, _ = blobs()
    vectors = vectors[:150]  # uneven group sizes
    labels = fit_topics(vectors, PARAMS).labels
    sizes = np.bincount(labels[labels >= 0])
    assert list(sizes) == sorted(sizes, reverse=True)


def test_same_seed_same_topics(blobs):
    vectors, _ = blobs()
    first = fit_topics(vectors, PARAMS)
    second = fit_topics(vectors, PARAMS)
    assert np.array_equal(first.labels, second.labels)
    assert np.allclose(first.coords, second.coords)
