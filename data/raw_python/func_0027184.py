def create_engine(
        database,
        minsize=1,
        maxsize=10,
        loop=None,
        dialect=_dialect,
        paramstyle=None,
        **kwargs):
    """
    A coroutine for Engine creation.

    Returns Engine instance with embedded connection pool.

    The pool has *minsize* opened connections to sqlite3.
    """
    coro = _create_engine(
        database=database,
        minsize=minsize,
        maxsize=maxsize,
        loop=loop,
        dialect=dialect,
        paramstyle=paramstyle,
        **kwargs
    )
    return _EngineContextManager(coro)