def do_sqlite_connect(dbapi_connection, connection_record):
    """Ensure SQLite checks foreign key constraints.

    For further details see "Foreign key support" sections on
    https://docs.sqlalchemy.org/en/latest/dialects/sqlite.html#foreign-key-support
    """
    # Enable foreign key constraint checking
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA foreign_keys=ON')
    cursor.close()