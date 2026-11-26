"""Command-line entry point: `arxiv-atlas <command>`."""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Annotated

import psycopg
import typer

from arxiv_atlas.db.migrate import migrate as apply_migrations
from arxiv_atlas.embeddings.pipeline import embed_pending
from arxiv_atlas.embeddings.store import EmbeddingStore
from arxiv_atlas.ingestion.harvest import EARLIEST_DATESTAMP, TARGET_SETS, Harvester
from arxiv_atlas.ingestion.oai_client import OaiClient
from arxiv_atlas.ingestion.repository import Repository
from arxiv_atlas.topics.fit import FitParams
from arxiv_atlas.topics.store import TopicRun, TopicStore

DEFAULT_DSN = "postgresql://atlas:atlas@localhost:5432/atlas"
# Trends are measured from here; older papers are in the database only because arXiv
# changed their metadata later.
TOPICS_SINCE = date(2020, 1, 1)

app = typer.Typer(no_args_is_help=True, add_completion=False)

Day = Annotated[date, typer.Option(parser=date.fromisoformat, metavar="YYYY-MM-DD")]
OptionalDay = Annotated[date | None, typer.Option(parser=date.fromisoformat, metavar="YYYY-MM-DD")]
Sets = Annotated[list[str] | None, typer.Option(help="OAI set spec, repeatable [default: all]")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Topics and trends in arXiv machine-learning papers."""
    # Libraries only report problems; our own modules also report progress.
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("arxiv_atlas").setLevel(logging.DEBUG if verbose else logging.INFO)


@app.command()
def migrate() -> None:
    """Create or upgrade the database schema."""
    with _db() as conn:
        applied = apply_migrations(conn)
    print(f"applied: {', '.join(applied)}" if applied else "schema is up to date")


@app.command()
def status() -> None:
    """Show what is in the database."""
    with _db() as conn:
        s = Repository(conn).status()
    print(
        f"papers: {s.total_papers} ({s.deleted_papers} deleted), "
        f"published {_day(s.first_published)} .. {_day(s.last_published)}\n"
        "\nby primary category:"
    )
    for category, count in s.by_primary_category:
        print(f"  {category:<12} {count}")
    print("\nharvest runs:")
    for set_spec, run_status, count, until in s.runs:
        print(f"  {set_spec:<14} {run_status:<8} {count:>5} windows, up to {until}")


@app.command()
def backfill(
    start: Annotated[
        date, typer.Option("--from", parser=date.fromisoformat, metavar="YYYY-MM-DD")
    ] = EARLIEST_DATESTAMP,
    until: Annotated[
        date | None,
        typer.Option(parser=date.fromisoformat, metavar="YYYY-MM-DD", show_default="today"),
    ] = None,
    window_days: int = 30,
    sets: Sets = None,
) -> None:
    """Harvest history from arXiv in datestamp windows; rerun to resume."""
    with _db() as conn, _client() as client:
        harvester = Harvester(client, Repository(conn), sets=sets or list(TARGET_SETS))
        harvester.backfill(start, until or harvester.today(), window_days)


@app.command()
def update(sets: Sets = None) -> None:
    """Harvest what arXiv changed since the last run."""
    with _db() as conn, _client() as client:
        Harvester(client, Repository(conn), sets=sets or list(TARGET_SETS)).update()


@app.command()
def embed(
    since: OptionalDay = None,
    batch_size: Annotated[int, typer.Option(help="lower it if the GPU runs out of memory")] = 32,
) -> None:
    """Embed papers that have no up-to-date vector, optionally only recent ones."""
    from arxiv_atlas.embeddings.model import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(batch_size=batch_size)
    with _db() as conn:
        print(f"embedded {embed_pending(EmbeddingStore(conn), embedder, since=since)} papers")


@app.command()
def search(query: str, k: Annotated[int, typer.Option("-k")] = 10) -> None:
    """Find papers by meaning, not keywords."""
    from arxiv_atlas.embeddings.model import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    with _db() as conn:
        hits = EmbeddingStore(conn).search(embedder.name, embedder.embed_query(query), k)
    for hit in hits:
        print(
            f"{hit.similarity:.3f}  {_day(hit.published_at)}  {hit.arxiv_id:<12} "
            f"{hit.primary_category:<8} {hit.title}"
        )


@app.command()
def fit_topics(
    since: Day = TOPICS_SINCE,
    sample_size: Annotated[int, typer.Option(help="papers to fit on; bounded by RAM")] = 100_000,
    min_cluster_size: int = FitParams().min_cluster_size,
    min_samples: int = FitParams().min_samples,
    seed: int = FitParams().seed,
) -> None:
    """Find topics on a sample of papers, then give every other paper one."""
    from arxiv_atlas.embeddings.model import MODEL_NAME
    from arxiv_atlas.topics.pipeline import assign_pending, fit_run

    params = FitParams(min_cluster_size=min_cluster_size, min_samples=min_samples, seed=seed)
    with _db() as conn:
        store = TopicStore(conn)
        run = _run(store, fit_run(store, MODEL_NAME, since, sample_size, params))
        assign_pending(store, run)
        _print_topics(store, run, limit=30)


@app.command()
def assign_topics() -> None:
    """Give papers embedded since the last fit a topic in the latest run."""
    from arxiv_atlas.topics.pipeline import assign_pending

    with _db() as conn:
        store = TopicStore(conn)
        run = _run(store, None)
        print(f"assigned {assign_pending(store, run)} papers to run {run.id}")


@app.command()
def topics(
    run: Annotated[int | None, typer.Option(show_default="latest")] = None,
    n: Annotated[int, typer.Option("-n", help="how many topics to show")] = 30,
) -> None:
    """List the largest topics of a run with their keywords."""
    with _db() as conn:
        store = TopicStore(conn)
        _print_topics(store, _run(store, run), n)


@contextmanager
def _db() -> Generator[psycopg.Connection]:
    with psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_DSN), autocommit=True) as conn:
        yield conn


@contextmanager
def _client() -> Generator[OaiClient]:
    with OaiClient(contact=os.environ.get("ARXIV_ATLAS_CONTACT") or None) as client:
        yield client


def _run(store: TopicStore, run_id: int | None) -> TopicRun:
    run = store.run(run_id)
    if run is None:
        raise SystemExit("no topic run found: run `arxiv-atlas fit-topics` first")
    return run


def _print_topics(store: TopicStore, run: TopicRun, limit: int) -> None:
    summary = store.summary(run.id)
    total = sum(t.papers for t in summary)
    unassigned = sum(t.papers for t in summary if t.topic_id is None)
    found = [t for t in summary if t.topic_id is not None]
    print(
        f"run {run.id}: {len(found)} topics over {total} papers, "
        f"{unassigned / max(total, 1):.0%} fit no topic\n\n{'topic':>5}  {'papers':>6}  keywords"
    )
    for topic in found[:limit]:
        print(f"{topic.topic_id:>5}  {topic.papers:>6}  {', '.join(topic.keywords[:8])}")


def _day(value: datetime | None) -> str:
    return value.astimezone(UTC).date().isoformat() if value else "-"
