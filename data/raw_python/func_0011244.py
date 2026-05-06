def is_connection_dropped(conn):  # Platform-specific
    """
    Returns True if the connection is dropped and should be closed.

    :param conn:
        :class:`httplib.HTTPConnection` object.

    Note: For platforms like AppEngine, this will always return ``False`` to
    let the platform handle connection recycling transparently for us.
    """
    sock = getattr(conn, 'sock', False)
    if sock is False:  # Platform-specific: AppEngine
        return False
    if sock is None:  # Connection already closed (such as by httplib).
        return True

    if not poll:
        if not select:  # Platform-specific: AppEngine
            return False

        try:
            return select([sock], [], [], 0.0)[0]
        except socket.error:
            return True

    # This version is better on platforms that support it.
    p = poll()
    p.register(sock, POLLIN)
    for (fno, ev) in p.poll(0.0):
        if fno == sock.fileno():
            # Either data is buffered (bad), or the connection is dropped.
            return True