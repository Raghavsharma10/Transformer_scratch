def _before_flush_handler(session, _flush_context, _instances):
    """Update version ID for all dirty, modified rows"""
    dialect = get_dialect(session)
    for row in session.dirty:
        if isinstance(row, SavageModelMixin) and is_modified(row, dialect):
            # Update row version_id
            row.update_version_id()