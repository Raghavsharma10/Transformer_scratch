def _make_hostport(conn, default_host, default_port, default_user='', default_password=None):
    """Convert a '[user[:pass]@]host:port' string to a Connection tuple.

    If the given connection is empty, use defaults.
    If no port is given, use the default.

    Args:
        conn (str): the string describing the target hsot/port
        default_host (str): the host to use if ``conn`` is empty
        default_port (int): the port to use if not given in ``conn``.

    Returns:
        (str, int): a (host, port) tuple.
    """
    parsed = urllib.parse.urlparse('//%s' % conn)
    return Connection(
        parsed.hostname or default_host,
        parsed.port or default_port,
        parsed.username if parsed.username is not None else default_user,
        parsed.password if parsed.password is not None else default_password,
    )