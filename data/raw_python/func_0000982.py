def connect(url, max_retries=None, **kwargs):
    """Connects to a Phoenix query server.

    :param url:
        URL to the Phoenix query server, e.g. ``http://localhost:8765/``

    :param autocommit:
        Switch the connection to autocommit mode.

    :param readonly:
        Switch the connection to readonly mode.

    :param max_retries:
        The maximum number of retries in case there is a connection error.

    :param cursor_factory:
        If specified, the connection's :attr:`~phoenixdb.connection.Connection.cursor_factory` is set to it.

    :returns:
        :class:`~phoenixdb.connection.Connection` object.
    """
    client = AvaticaClient(url, max_retries=max_retries)
    client.connect()
    return Connection(client, **kwargs)