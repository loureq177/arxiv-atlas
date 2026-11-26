"""Command-line entry point: `arxiv-atlas <command>`."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, date, datetime

import psycopg

from arxiv_atlas.db.migrate import migrate
from arxiv_atlas.embeddings.pipeline import embed_pending
from arxiv_atlas.embeddings.store import EmbeddingStore
from arxiv_atlas.ingestion.harvest import EARLIEST_DATESTAMP, TARGET_SETS, Harvester
from arxiv_atlas.ingestion.oai_client import OaiClient
from arxiv_atlas.ingestion.repository import DatabaseStatus, Repository

DEFAULT_DSN = "postgresql://atlas:atlas@localhost:5432/atlas"


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    # Libraries only report problems; our own modules also report progress.
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("arxiv_atlas").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    with psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_DSN), autocommit=True) as conn:
        args.handler(conn, args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arxiv-atlas")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="create or upgrade the database schema").set_defaults(
        handler=_migrate
    )
    sub.add_parser("status", help="show what is in the database").set_defaults(handler=_status)

    backfill = sub.add_parser("backfill", help="harvest history in datestamp windows")
    backfill.add_argument(
        "--from", dest="start", type=date.fromisoformat, default=EARLIEST_DATESTAMP
    )
    backfill.add_argument(
        "--until", dest="end", type=date.fromisoformat, default=None, help="default: today (UTC)"
    )
    backfill.add_argument("--window-days", type=int, default=30)
    _add_sets_argument(backfill)
    backfill.set_defaults(handler=_harvest)

    update = sub.add_parser("update", help="incremental harvest since the last run")
    _add_sets_argument(update)
    update.set_defaults(handler=_harvest)

    embed = sub.add_parser("embed", help="embed papers that have no up-to-date vector")
    embed.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        help="only papers published on or after this date (default: all)",
    )
    embed.add_argument("--batch-size", type=int, default=32, help="lower it if the GPU runs out")
    embed.set_defaults(handler=_embed)

    search = sub.add_parser("search", help="find papers by meaning, not keywords")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=10, help="number of results")
    search.set_defaults(handler=_search)
    return parser


def _add_sets_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sets",
        nargs="+",
        default=list(TARGET_SETS),
        metavar="SET",
        help="OAI set specs (default: %(default)s)",
    )


def _migrate(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    applied = migrate(conn)
    print(f"applied: {', '.join(applied)}" if applied else "schema is up to date")


def _status(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    _print_status(Repository(conn).status())


def _harvest(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    with OaiClient(contact=os.environ.get("ARXIV_ATLAS_CONTACT") or None) as client:
        harvester = Harvester(client, Repository(conn), sets=args.sets)
        if args.command == "backfill":
            harvester.backfill(args.start, args.end or harvester.today(), args.window_days)
        else:
            harvester.update()


def _embed(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    from arxiv_atlas.embeddings.model import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(batch_size=args.batch_size)
    embedded = embed_pending(EmbeddingStore(conn), embedder, since=args.since)
    print(f"embedded {embedded} papers")


def _search(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    from arxiv_atlas.embeddings.model import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    hits = EmbeddingStore(conn).search(embedder.name, embedder.embed_query(args.query), args.k)
    for hit in hits:
        print(
            f"{hit.similarity:.3f}  {_d(hit.published_at)}  {hit.arxiv_id:<12} "
            f"{hit.primary_category:<8} {hit.title}"
        )


def _print_status(status: DatabaseStatus) -> None:
    print(
        f"papers: {status.total_papers} ({status.deleted_papers} deleted), "
        f"published {_d(status.first_published)} .. {_d(status.last_published)}"
    )
    print("\nby primary category:")
    for category, count in status.by_primary_category:
        print(f"  {category:<12} {count}")
    print("\nharvest runs:")
    for set_spec, run_status, count, until in status.runs:
        print(f"  {set_spec:<14} {run_status:<8} {count:>5} windows, up to {until}")


def _d(value: datetime | None) -> str:
    return value.astimezone(UTC).date().isoformat() if value else "-"


if __name__ == "__main__":
    main()
