def wrap(stream, unicode=False, window=1024, echo=False, close_stream=True):
    """Wrap a stream to implement expect functionality.

    This function provides a convenient way to wrap any Python stream (a
    file-like object) or socket with an appropriate :class:`Expecter` class for
    the stream type. The returned object adds an :func:`Expect.expect` method
    to the stream, while passing normal stream functions like *read*/*recv*
    and *write*/*send* through to the underlying stream.

    Here's an example of opening and wrapping a pair of network sockets::

        import socket
        import streamexpect

        source, drain = socket.socketpair()
        expecter = streamexpect.wrap(drain)
        source.sendall(b'this is a test')
        match = expecter.expect_bytes(b'test', timeout=5)

        assert match is not None

    :param stream: The stream/socket to wrap.
    :param bool unicode: If ``True``, the wrapper will be configured for
        Unicode matching, otherwise matching will be done on binary.
    :param int window: Historical characters to buffer.
    :param bool echo: If ``True``, echoes received characters to stdout.
    :param bool close_stream: If ``True``, and the wrapper is used as a context
        manager, closes the stream at the end of the context manager.
    """
    if hasattr(stream, 'read'):
        proxy = PollingStreamAdapter(stream)
    elif hasattr(stream, 'recv'):
        proxy = PollingSocketStreamAdapter(stream)
    else:
        raise TypeError('stream must have either read or recv method')

    if echo and unicode:
        callback = _echo_text
    elif echo and not unicode:
        callback = _echo_bytes
    else:
        callback = None

    if unicode:
        expecter = TextExpecter(proxy, input_callback=callback, window=window,
                                close_adapter=close_stream)
    else:
        expecter = BytesExpecter(proxy, input_callback=callback, window=window,
                                 close_adapter=close_stream)

    return expecter