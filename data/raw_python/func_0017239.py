def drop_alembic_version_table():
    """Drop alembic_version table."""
    if _db.engine.dialect.has_table(_db.engine, 'alembic_version'):
        alembic_version = _db.Table('alembic_version', _db.metadata,
                                    autoload_with=_db.engine)
        alembic_version.drop(bind=_db.engine)