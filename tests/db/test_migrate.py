from importlib.resources import files

from arxiv_atlas.db.migrate import migrate

MIGRATIONS = files("arxiv_atlas.db").joinpath("migrations")
ALL_VERSIONS = sorted(
    f.name.removesuffix(".sql") for f in MIGRATIONS.iterdir() if f.name.endswith(".sql")
)


def test_applies_every_migration_once(db):
    # The fixture already migrated, so a second run has nothing to do.
    assert migrate(db) == []
    versions = [row[0] for row in db.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    assert versions == ALL_VERSIONS


def test_upgrades_a_database_created_before_the_runner(db):
    # The old `init-db` ran the first file by hand and recorded nothing.
    db.execute("DROP SCHEMA public CASCADE")
    db.execute("CREATE SCHEMA public")
    db.execute(MIGRATIONS.joinpath("0001_ingestion.sql").read_bytes())
    assert migrate(db) == ALL_VERSIONS
