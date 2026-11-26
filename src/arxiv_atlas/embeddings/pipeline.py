"""Embed every paper that has no up-to-date vector. Safe to interrupt and rerun."""

from __future__ import annotations

import logging
import time
from datetime import date

from arxiv_atlas.embeddings.model import Embedder
from arxiv_atlas.embeddings.store import EmbeddingStore

log = logging.getLogger(__name__)


def embed_pending(
    store: EmbeddingStore, embedder: Embedder, since: date | None = None, chunk_size: int = 1024
) -> int:
    """Embed in chunks, each committed on its own, and return how many were embedded.

    An interrupted run loses at most the chunk in progress; the next run finds the rest
    still pending.
    """
    total = store.count_pending(embedder.name, since)
    log.info("%d papers to embed with %s", total, embedder.name)
    done, after, started = 0, "", time.monotonic()
    while batch := store.pending(embedder.name, since, after, chunk_size):
        store.save(embedder.name, batch, embedder.embed_documents([p.text for p in batch]))
        done += len(batch)
        after = batch[-1].arxiv_id
        rate = done / (time.monotonic() - started)
        log.info(
            "%d/%d embedded, %.0f papers/s, %.0f min left",
            done,
            total,
            rate,
            max(total - done, 0) / rate / 60,
        )
    return done
