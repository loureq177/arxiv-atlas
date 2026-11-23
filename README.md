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
docker compose up -d                 # PostgreSQL 17 + pgvector
cp .env.example .env                 # then `set -a; source .env; set +a`
uv sync
uv run arxiv-atlas init-db
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
30 6 * * * cd /path/to/projekt && set -a && . ./.env && uv run arxiv-atlas update >> data/update.log 2>&1
```

### Tests

```sh
uv run pytest                                              # unit tests, no network/DB
TEST_DATABASE_URL=postgresql://.../atlas_test uv run pytest # + repository tests
```

The repository tests drop and recreate tables: use a separate database.
