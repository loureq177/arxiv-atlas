"""Topic runs end to end against a real PostgreSQL; see tests/conftest.py."""

from datetime import date

import pytest
from pgvector import HalfVector

from arxiv_atlas.topics.fit import FitParams
from arxiv_atlas.topics.pipeline import assign_pending, fit_run
from arxiv_atlas.topics.store import TopicStore

MODEL = "fake-model"
WORDS = ["graph neural message", "diffusion image denoising", "speech audio speaker"]


@pytest.fixture
def store(db, blobs):
    vectors, groups = blobs()
    with db.cursor() as cur:
        for i, group in enumerate(groups):
            arxiv_id = f"2610.{i:05d}"
            published = "2019-12-01" if i == 0 else "2026-10-01"
            cur.execute(
                """INSERT INTO papers (arxiv_id, title, abstract, authors, primary_category,
                                       categories, published_at, updated_at, num_versions,
                                       oai_datestamp)
                   VALUES (%s, %s, %s, 'A', 'cs.LG', ARRAY['cs.LG'], %s, %s, 1, %s)""",
                (arxiv_id, WORDS[group], WORDS[group], published, published, published),
            )
    store = TopicStore(db)
    with db.cursor() as cur:
        for i, vector in enumerate(vectors):
            cur.execute(
                "INSERT INTO paper_embeddings (arxiv_id, model, text_md5, embedding)"
                " VALUES (%s, %s, '', %s)",
                (f"2610.{i:05d}", MODEL, HalfVector(vector)),
            )
    return store


def fit(store, sample_size=120):
    params = FitParams(n_neighbors=10, min_cluster_size=20, min_samples=5)
    return store.run(fit_run(store, MODEL, date(2020, 1, 1), sample_size, params))


def test_fit_then_assign_covers_every_paper_in_scope(store):
    run = fit(store)
    assert assign_pending(store, run) == 179 - 120  # 180 papers, one published before 2020
    assert assign_pending(store, run) == 0

    summary = store.summary(run.id)
    assert sum(t.papers for t in summary) == 179
    topics = [t for t in summary if t.topic_id is not None]
    assert len(topics) == 3
    assert {t.keywords[0] for t in topics} <= {w for ws in WORDS for w in ws.split()}


def test_runs_are_independent(store):
    first = fit(store)
    second = fit(store, sample_size=100)
    assert second.id != first.id
    assert store.run().id == second.id  # latest by default
    assert sum(t.papers for t in store.summary(first.id)) == 120


def test_too_few_papers(store):
    with pytest.raises(ValueError, match="embed more"):
        fit_run(store, "other-model", date(2020, 1, 1), 100, FitParams())
