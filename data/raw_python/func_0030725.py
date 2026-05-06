def _validate_version(connection, dsn):
    """ Performs on-the-fly schema updates based on the models version.

    Raises:
        DatabaseError: if user uses old sqlite database.

    """
    try:
        version = get_stored_version(connection)
    except VersionIsNotStored:
        logger.debug('Version not stored in the db: assuming new database creation.')
        version = SCHEMA_VERSION
        _update_version(connection, version)
    assert isinstance(version, int)

    if version > 10 and version < 100:
        raise DatabaseError('You are trying to open an old SQLite database.')

    if _migration_required(connection):
        migrate(connection, dsn)