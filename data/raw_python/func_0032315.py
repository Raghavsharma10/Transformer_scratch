def connect_if_correct_version(db_path, version):
    """Return a sqlite3 database connection if the version in the database's
    metadata matches the version argument.

    Also implicitly checks for whether the data in this database has
    been completely filled, since we set the version last.

    TODO: Make an explicit 'complete' flag to the metadata.
    """
    db = Database(db_path)
    if db.has_version() and db.version() == version:
        return db.connection
    return None