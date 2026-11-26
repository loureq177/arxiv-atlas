"""Embedding pipeline and search against a real PostgreSQL; see tests/conftest.py."""

import zlib
from collections.abc import Sequence
from datetime import date

import numpy as np
import pytest

from arxiv_atlas.embeddings.model import DIM, Vectors
from arxiv_atlas.embeddings.pipeline import embed_pending
from arxiv_atlas.embeddings.store import EmbeddingStore


class BagOfWordsEmbedder:
    """Deterministic stand-in for the model: texts sharing words get similar vectors."""

    def __init__(self, name="fake-model"):
        self.name = name
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.calls.append(list(texts))
        return np.stack([self.embed_query(t) for t in texts])

    def embed_query(self, text: str) -> Vectors:
        vector = np.zeros(DIM, dtype=np.float32)
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % DIM] += 1.0
        return vector / np.linalg.norm(vector)


def add_paper(db, arxiv_id, title, abstract, published="2026-10-01"):
    db.execute(
        """INSERT INTO papers (arxiv_id, title, abstract, authors, primary_category,
                               categories, published_at, updated_at, num_versions,
                               oai_datestamp)
           VALUES (%s, %s, %s, 'A', 'cs.LG', ARRAY['cs.LG'], %s, %s, 1, %s)""",
        (arxiv_id, title, abstract, published, published, published),
    )


@pytest.fixture
def store(db):
    add_paper(db, "2610.00001", "Graph neural networks", "message passing on molecules")
    add_paper(db, "2610.00002", "Diffusion models", "denoising image generation")
    add_paper(db, "2001.00003", "Old paper", "support vector machines", published="2020-01-05")
    return EmbeddingStore(db)


def test_embeds_everything_once(store):
    embedder = BagOfWordsEmbedder()
    assert embed_pending(store, embedder, chunk_size=2) == 3
    assert len(embedder.calls) == 2  # chunks of 2 and 1
    assert embed_pending(store, embedder) == 0


def test_embeds_title_and_abstract(store):
    embedder = BagOfWordsEmbedder()
    embed_pending(store, embedder, since=date(2026, 10, 1))
    assert "Diffusion models\n\ndenoising image generation" in embedder.calls[0]


def test_since_skips_older_papers(store):
    assert embed_pending(store, BagOfWordsEmbedder(), since=date(2026, 1, 1)) == 2


def test_changed_abstract_is_embedded_again(db, store):
    embed_pending(store, BagOfWordsEmbedder())
    db.execute("UPDATE papers SET abstract = 'new abstract' WHERE arxiv_id = '2610.00002'")
    assert embed_pending(store, BagOfWordsEmbedder()) == 1


def test_switching_models_embeds_everything_again(store):
    embed_pending(store, BagOfWordsEmbedder("model-a"))
    assert embed_pending(store, BagOfWordsEmbedder("model-b")) == 3


def test_search_ranks_by_similarity(store):
    embedder = BagOfWordsEmbedder()
    embed_pending(store, embedder)
    hits = store.search(embedder.name, embedder.embed_query("graph neural networks"), k=2)
    assert [h.arxiv_id for h in hits] == ["2610.00001", "2610.00002"]
    assert hits[0].similarity > hits[1].similarity


def test_search_ignores_vectors_from_another_model(store):
    embed_pending(store, BagOfWordsEmbedder("model-a"))
    query = BagOfWordsEmbedder().embed_query("graph")
    assert store.search("model-b", query, k=5) == []


def test_save_rejects_mismatched_lengths(store):
    pending = store.pending("fake-model", None, "", 10)
    with pytest.raises(ValueError, match="3 papers but 1 vectors"):
        store.save("fake-model", pending, np.zeros((1, DIM), dtype=np.float32))
