def _read_msg(session):
    """
    Perform a read on input socket to consume message and then return the
    payload and block_id in a tuple.

    :param session: Push Session to read data for.
    """
    if len(session.data) == session.message_length:
        # Data Already completely read.  Return
        return True

    try:
        data = session.socket.recv(session.message_length - len(session.data))
        if len(data) == 0:
            raise PushException("No Data on Socket!")
        session.data += data
    except ssl.SSLError:
        # This can happen when select gets triggered
        # for an SSL socket and data has not yet been
        # read.  Wait for it to get triggered again.
        return False

    # Whether or not all data was read.
    return len(session.data) == session.message_length