-- One vector per paper, computed from its title and abstract.
CREATE TABLE paper_embeddings (
    arxiv_id   text PRIMARY KEY REFERENCES papers (arxiv_id) ON DELETE CASCADE,
    model      text NOT NULL,             -- vectors from different models are not comparable
    text_md5   text NOT NULL,             -- of the embedded text: a new abstract re-embeds
    -- Half precision halves the storage; ranking quality is unchanged in practice.
    -- 1024 = output size of the model in arxiv_atlas.embeddings.model.
    embedding  halfvec(1024) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Approximate nearest-neighbour index: top-k by cosine distance without scanning every row.
CREATE INDEX paper_embeddings_embedding_idx
    ON paper_embeddings USING hnsw (embedding halfvec_cosine_ops);
