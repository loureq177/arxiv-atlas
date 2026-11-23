"""Command-line entry point: `arxiv-atlas <command>`."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, date, datetime

from arxiv_atlas.ingestion.harvest import EARLIEST_DATESTAMP, TARGET_SETS, Harvester
from arxiv_atlas.ingestion.oai_client import OaiClient
from arxiv_atlas.ingestion.repository import Repository

DEFAULT_DSN = "postgresql://atlas:atlas@localhost:5432/atlas"


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    repo = Repository.connect(os.environ.get("DATABASE_URL", DEFAULT_DSN))
    try:
        if args.command == "init-db":
            repo.init_schema()
            print("schema ready")
        elif args.command == "status":
            _print_status(repo.status())
        else:
            with OaiClient(contact=os.environ.get("ARXIV_ATLAS_CONTACT") or None) as client:
                harvester = Harvester(client, repo, sets=args.sets)
                if args.command == "backfill":
                    harvester.backfill(args.start, args.end or harvester.today(), args.window_days)
                else:
                    harvester.update()
    finally:
        repo.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arxiv-atlas")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create tables and the pgvector extension")
    sub.add_parser("status", help="show what is in the database")

    sets_kwargs = dict(
        nargs="+", default=list(TARGET_SETS), metavar="SET",
        help="OAI set specs (default: %(default)s)",
    )
    backfill = sub.add_parser("backfill", help="harvest history in datestamp windows")
    backfill.add_argument("--from", dest="start", type=date.fromisoformat,
                          default=EARLIEST_DATESTAMP)
    backfill.add_argument("--until", dest="end", type=date.fromisoformat, default=None,
                          help="default: today (UTC)")
    backfill.add_argument("--window-days", type=int, default=30)
    backfill.add_argument("--sets", **sets_kwargs)

    update = sub.add_parser("update", help="incremental harvest since the last run")
    update.add_argument("--sets", **sets_kwargs)
    return parser


def _print_status(status: dict[str, object]) -> None:
    total, deleted, first, last = status["papers"]
    print(f"papers: {total} ({deleted} deleted), published {_d(first)} .. {_d(last)}")
    print("\nby primary category:")
    for category, count in status["by_primary_category"]:
        print(f"  {category:<12} {count}")
    print("\nharvest runs:")
    for set_spec, run_status, count, until in status["runs"]:
        print(f"  {set_spec:<14} {run_status:<8} {count:>5} windows, up to {until}")


def _d(value: datetime | None) -> str:
    return value.astimezone(UTC).date().isoformat() if value else "-"


if __name__ == "__main__":
    main()
