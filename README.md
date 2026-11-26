# ArXiv Atlas

Fine-grained topic discovery and trend detection for arXiv ML papers
(cs.LG, cs.CL, cs.AI, stat.ML).

## Data ingestion

Metadata is harvested from arXiv's [OAI-PMH endpoint](https://oaipmh.arxiv.org/oai)
in the `arXivRaw` format (the only format with per-version dates, so `published_at`
is the real v1 date) and stored in PostgreSQL.

- One request every ≥3 s, retries with `Retry-After` / exponential backoff.
- Work is split into datestamp windows; each window is a row in `harvest_runs` that
  stores the current resumption token, so a crash resumes from the last saved page.
- Each page is written in one transaction together with its checkpoint.
- Upserts are idempotent and never overwrite newer metadata with older.

### Setup

```sh
podman compose up -d                 # PostgreSQL 18 + pgvector (docker compose works too)
cp .env.example .env                 # then `set -a; source .env; set +a`
uv sync
uv run arxiv-atlas migrate             # create or upgrade the schema
```

Without Docker, PostgreSQL with the `pgvector` extension installed works the same;
point `DATABASE_URL` at it.

### Usage

```sh
uv run arxiv-atlas backfill                      # full history, 30-day windows
uv run arxiv-atlas backfill --from 2024-01-01    # or a slice
uv run arxiv-atlas update                        # daily incremental harvest
uv run arxiv-atlas status
```

`backfill` can be interrupted and rerun at any time: finished windows are skipped
and unfinished ones resume. `update` refuses to run for sets that were never
backfilled.

Daily run via cron:

```cron
30 6 * * * cd /path/to/arxiv-atlas && set -a && . ./.env && uv run arxiv-atlas update >> data/update.log 2>&1
```

## Embeddings and search

Each paper's title and abstract become one 1024-dimensional vector
([Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), stored as
pgvector `halfvec` with an HNSW index). Papers about the same thing get nearby vectors,
whatever their wording.

```sh
uv run arxiv-atlas embed --since 2026-09-01     # a slice first: ~20 min on a 4 GB GPU
uv run arxiv-atlas embed                         # everything: hours; resumable
uv run arxiv-atlas search "LLM agents that use tools"
```

`embed` only processes papers without an up-to-date vector, so it is safe to stop and
rerun, and after an `update` it embeds just the new papers. A changed abstract is
embedded again.

## Tests

```sh
uv run pytest                                              # unit tests, no network/DB
TEST_DATABASE_URL=postgresql://.../atlas_test uv run pytest # + repository tests
```

The database tests drop the whole schema: use a separate database.
