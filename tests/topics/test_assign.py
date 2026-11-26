import numpy as np

from arxiv_atlas.topics.assign import Reference, assign_topics


def unit(*rows):
    vectors = np.array(rows, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


REFERENCE = Reference(
    vectors=unit([1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.1, 0.9, 0], [0, 0, 1]),
    labels=np.array([0, 0, 1, 1, -1]),
    coords=np.array([[0, 0], [2, 0], [10, 10], [12, 10], [50, 50]], dtype=np.float32),
)


def test_paper_joins_the_topic_of_its_neighbours():
    labels, coords = assign_topics(REFERENCE, unit([0.95, 0.05, 0]), k=2)
    assert labels.tolist() == [0]
    assert coords.tolist() == [[1.0, 0.0]]  # midway between the two voters


def test_closer_neighbours_weigh_more():
    reference = Reference(
        vectors=unit([0, 1, 0], [1, 0, 0], [1, 0.2, 0]),
        labels=np.array([1, 0, 0]),
        coords=np.zeros((3, 2), dtype=np.float32),
    )
    # One very similar paper of topic 1 outweighs two far ones of topic 0.
    labels, _ = assign_topics(reference, unit([0.3, 1, 0]), k=3)
    assert labels.tolist() == [1]


def test_paper_near_unassigned_papers_gets_no_topic():
    labels, coords = assign_topics(REFERENCE, unit([0, 0, 1]), k=1)
    assert labels.tolist() == [-1]
    assert coords.tolist() == [[50.0, 50.0]]


def test_k_larger_than_reference_and_chunking():
    queries = unit([1, 0, 0], [0, 1, 0], [1, 0.05, 0])
    labels, _ = assign_topics(REFERENCE, queries, k=50, chunk_size=2)
    assert labels.shape == (3,)
