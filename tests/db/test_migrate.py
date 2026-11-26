from arxiv_atlas.db.migrate import migrate


def test_applies_every_migration_once(db):
    # The fixture already migrated, so a second run has nothing to do.
    assert migrate(db) == []
    versions = [row[0] for row in db.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    assert versions == ["0001_ingestion", "0002_embeddings"]


def test_upgrades_a_database_created_before_the_runner(db):
    # The old `init-db` created the ingestion tables but recorded nothing.
    db.execute("DROP TABLE schema_migrations, paper_embeddings")
    assert migrate(db) == ["0001_ingestion", "0002_embeddings"]
