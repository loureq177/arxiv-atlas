"""The embedding model behind a small interface, so tests can swap in a fake."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# A commit on the Hugging Face Hub: a branch name could start serving other weights.
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DIM = 1024  # must match halfvec(1024) in migrations/0002_embeddings.sql
# The longest title + abstract we measured is ~500 tokens. The cap bounds GPU memory.
MAX_TOKENS = 512

type Vectors = NDArray[np.float32]


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vectors: ...


class SentenceTransformerEmbedder:
    """Unit-length vectors, so cosine similarity is a plain dot product."""

    def __init__(
        self, name: str = MODEL_NAME, revision: str = MODEL_REVISION, batch_size: int = 32
    ) -> None:
        # Imported here: torch takes seconds to import and most commands never need it.
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()  # type: ignore[no-untyped-call]

        gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        self.name = name
        self._batch_size = batch_size
        kwargs: dict[str, Any] = {
            "revision": revision,
            "model_kwargs": {"dtype": torch.float16 if gpu else torch.float32},
        }
        try:
            # Cache first: no network needed once the weights are downloaded.
            self._model = SentenceTransformer(name, local_files_only=True, **kwargs)
        except OSError:
            self._model = SentenceTransformer(name, **kwargs)
        self._model.max_seq_length = MAX_TOKENS
        dim = self._model.get_embedding_dimension()
        if dim != DIM:
            raise ValueError(f"{name} produces {dim}-d vectors, the database stores {DIM}-d")

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        vectors = self._model.encode_document(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,  # the pipeline logs progress per chunk
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> Vectors:
        # Qwen3-Embedding prefixes queries with a task instruction and documents with
        # nothing; encode_query/encode_document apply the prompts from the model config.
        vector = self._model.encode_query(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)
