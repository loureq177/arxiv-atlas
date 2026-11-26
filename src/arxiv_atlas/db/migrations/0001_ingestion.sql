-- Idempotent (IF NOT EXISTS): databases created before the migration runner already
-- have these tables, and the runner applies this file to them once more.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id         text PRIMARY KEY,          -- '2301.01234' or old-style 'hep-th/9901001'
    title            text NOT NULL,
    abstract         text NOT NULL,
    authors          text NOT NULL,             -- raw arXivRaw string: 'A and B and C'
    submitter        text,
    primary_category text NOT NULL,             -- first entry of arXiv's category list
    categories       text[] NOT NULL,           -- primary + cross-lists, primary first
    published_at     timestamptz NOT NULL,      -- date of v1: use this for trends
    updated_at       timestamptz NOT NULL,      -- date of the latest version
    num_versions     integer NOT NULL,
    comments         text,
    journal_ref      text,
    doi              text,
    license          text,
    oai_datestamp    date NOT NULL,             -- last metadata change on arXiv's side
    is_deleted       boolean NOT NULL DEFAULT false,
    first_seen_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS papers_published_at_idx ON papers (published_at);
CREATE INDEX IF NOT EXISTS papers_primary_category_idx ON papers (primary_category);
-- Supports `categories @> ARRAY['cs.CL']` and `categories && ARRAY[...]`.
CREATE INDEX IF NOT EXISTS papers_categories_idx ON papers USING gin (categories);

-- One row per (set, datestamp window). Doubles as the checkpoint for resuming.
CREATE TABLE IF NOT EXISTS harvest_runs (
    id                bigserial PRIMARY KEY,
    set_spec          text NOT NULL,
    from_date         date NOT NULL,
    until_date        date NOT NULL,
    status            text NOT NULL CHECK (status IN ('running', 'done', 'failed')),
    resumption_token  text,
    token_expires_at  timestamptz,
    pages_fetched     integer NOT NULL DEFAULT 0,
    records_seen      integer NOT NULL DEFAULT 0,
    records_upserted  integer NOT NULL DEFAULT 0,
    started_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz,
    error             text
);

CREATE INDEX IF NOT EXISTS harvest_runs_window_idx
    ON harvest_runs (set_spec, from_date, until_date);
