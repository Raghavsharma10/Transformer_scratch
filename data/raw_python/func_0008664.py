def connect_and_open_channel(host='localhost',
                             port=5672,
                             username='guest', password='guest',
                             virtual_host='/',
                             on_connection_close=None, *,
                             loop=None, **kwargs):
    """
    Connect to an AMQP server and open a channel on the connection.
    This function is a :ref:`coroutine <coroutine>`.

    Parameters of this function are the same as :func:`connect`.

    :return: a tuple of ``(connection, channel)``.

    Equivalent to::

        connection = yield from connect(host, port, username, password, virtual_host, on_connection_close, loop=loop, **kwargs)
        channel = yield from connection.open_channel()
        return connection, channel
    """
    connection = yield from connect(host, port, username, password, virtual_host, on_connection_close, loop=loop, **kwargs)
    channel = yield from connection.open_channel()
    return connection, channel