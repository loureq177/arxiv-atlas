-- One row per fit of the topic model. Topic ids only mean something within their run:
-- refitting produces a new run rather than renumbering the old one.
CREATE TABLE topic_runs (
    id            bigserial PRIMARY KEY,
    model         text NOT NULL,          -- embedding model the vectors came from
    published_since date NOT NULL,        -- papers published before this are left out
    params        jsonb NOT NULL,         -- every setting needed to repeat the fit
    sample_size   integer NOT NULL,       -- papers the topics were fitted on
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE topics (
    run_id    bigint NOT NULL REFERENCES topic_runs (id) ON DELETE CASCADE,
    topic_id  integer NOT NULL,           -- 0 is the largest topic of the run
    size      integer NOT NULL,           -- papers of the fitting sample in this topic
    keywords  text[] NOT NULL,            -- most distinctive first
    label     text,                       -- human-readable name, filled in later
    PRIMARY KEY (run_id, topic_id)
);

CREATE TABLE paper_topics (
    run_id    bigint NOT NULL REFERENCES topic_runs (id) ON DELETE CASCADE,
    arxiv_id  text NOT NULL REFERENCES papers (arxiv_id) ON DELETE CASCADE,
    topic_id  integer,                    -- NULL: the paper fits no topic
    x         real NOT NULL,              -- position on the 2D map
    y         real NOT NULL,
    fitted    boolean NOT NULL,           -- true: in the fitting sample; false: assigned later
    PRIMARY KEY (run_id, arxiv_id),
    FOREIGN KEY (run_id, topic_id) REFERENCES topics (run_id, topic_id)
);

CREATE INDEX paper_topics_topic_idx ON paper_topics (run_id, topic_id);
