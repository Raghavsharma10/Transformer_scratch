def _connect(dbfile: 'PathLike') -> apsw.Connection:
    """Connect to SQLite database file."""
    conn = apsw.Connection(os.fspath(dbfile))
    _set_foreign_keys(conn, 1)
    assert _get_foreign_keys(conn) == 1
    return conn