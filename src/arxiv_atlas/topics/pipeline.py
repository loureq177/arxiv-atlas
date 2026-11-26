"""Fit a topic run on a sample of papers, then give every other paper a topic."""

from __future__ import annotations

import dataclasses
import logging
from datetime import date

import numpy as np

from arxiv_atlas.topics.assign import assign_topics
from arxiv_atlas.topics.fit import FitParams, fit_topics
from arxiv_atlas.topics.keywords import topic_keywords
from arxiv_atlas.topics.store import TopicRun, TopicStore

log = logging.getLogger(__name__)


def fit_run(store: TopicStore, model: str, since: date, sample_size: int, params: FitParams) -> int:
    """Fit topics on a sample and store them as a new run; returns the run id.

    The sample keeps memory bounded: UMAP on all papers would not fit in a laptop's RAM.
    The rest of the papers get their topic from `assign_pending`.
    """
    papers = store.sample(model, since, sample_size, params.seed)
    if len(papers.ids) < params.min_cluster_size * 2:
        raise ValueError(f"only {len(papers.ids)} embedded papers since {since}: embed more")
    log.info("fitting topics on %d papers", len(papers.ids))
    result = fit_topics(papers.vectors, params)
    keywords = topic_keywords(papers.texts, result.labels)
    run_id = store.save_fit(model, since, dataclasses.asdict(params), papers, result, keywords)
    log.info(
        "run %d: %d topics, %.0f%% of the sample in none",
        run_id,
        len(keywords),
        100 * np.mean(result.labels < 0),
    )
    return run_id


def assign_pending(store: TopicStore, run: TopicRun, chunk_size: int = 20_000) -> int:
    """Give every in-scope paper without a topic in `run` one; returns how many."""
    reference = store.reference(run.id)
    done, after = 0, ""
    while (batch := store.unassigned(run, after, chunk_size)).ids:
        labels, coords = assign_topics(reference, batch.vectors)
        store.save_assigned(run.id, batch.ids, labels, coords)
        done += len(batch.ids)
        after = batch.ids[-1]
        log.info("%d papers assigned", done)
    return done
