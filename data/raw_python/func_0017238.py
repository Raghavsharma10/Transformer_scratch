def create_alembic_version_table():
    """Create alembic_version table."""
    alembic = current_app.extensions['invenio-db'].alembic
    if not alembic.migration_context._has_version_table():
        alembic.migration_context._ensure_version_table()
        for head in alembic.script_directory.revision_map._real_heads:
            alembic.migration_context.stamp(alembic.script_directory, head)