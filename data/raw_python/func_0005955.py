def send(message, request_context=None, binary=False):
    """Sends a message to websocket.

    :param str message: data to send

    :param request_context:

    :raises IOError: If unable to send a message.
    """
    if binary:
        return uwsgi.websocket_send_binary(message, request_context)

    return uwsgi.websocket_send(message, request_context)