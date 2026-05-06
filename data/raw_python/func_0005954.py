def recv(request_context=None, non_blocking=False):
    """Receives data from websocket.

    :param request_context:

    :param bool non_blocking:

    :rtype: bytes|str

    :raises IOError: If unable to receive a message.
    """

    if non_blocking:
        result = uwsgi.websocket_recv_nb(request_context)

    else:
        result = uwsgi.websocket_recv(request_context)

    return result