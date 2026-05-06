def patch_connection(filename=':memory:'):
    """
    ``filename``: rlite filename to store db in, or memory
    Patch the redis-py Connection and the
    static from_url() of Redis and StrictRedis to use RliteConnection
    """

    if no_redis:
        raise Exception("redis package not found, please install redis-py via 'pip install redis'")

    RliteConnection.set_file(filename)

    global orig_classes

    # already patched
    if orig_classes:
        return

    orig_classes = (redis.connection.Connection,
                    redis.connection.ConnectionPool)

    _set_classes(RliteConnection, RliteConnectionPool)