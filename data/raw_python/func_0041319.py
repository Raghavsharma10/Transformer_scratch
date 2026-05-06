def _after_flush_handler(session, _flush_context):
    """Archive all new/updated/deleted data"""
    dialect = get_dialect(session)
    handlers = [
        (_versioned_delete, session.deleted),
        (_versioned_insert, session.new),
        (_versioned_update, session.dirty),
    ]
    for handler, rows in handlers:
        # TODO: Bulk archive insert statements
        for row in rows:
            if not isinstance(row, SavageModelMixin):
                continue
            if not hasattr(row, 'ArchiveTable'):
                raise LogTableCreationError('Need to register Savage tables!!')
            user_id = getattr(row, '_updated_by', None)
            handler(row, session, user_id, dialect)