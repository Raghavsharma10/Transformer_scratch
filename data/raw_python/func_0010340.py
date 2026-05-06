def _read_msg_header(session):
    """
    Perform a read on input socket to consume headers and then return
    a tuple of message type, message length.

    :param session: Push Session to read data for.

    Returns response type (i.e. PUBLISH_MESSAGE) if header was completely
    read, otherwise None if header was not completely read.
    """
    try:
        data = session.socket.recv(6 - len(session.data))
        if len(data) == 0:  # No Data on Socket. Likely closed.
            return NO_DATA
        session.data += data
        # Data still not completely read.
        if len(session.data) < 6:
            return INCOMPLETE

    except ssl.SSLError:
        # This can happen when select gets triggered
        # for an SSL socket and data has not yet been
        # read.
        return INCOMPLETE

    session.message_length = struct.unpack('!i', session.data[2:6])[0]
    response_type = struct.unpack('!H', session.data[0:2])[0]

    # Clear out session data as header is consumed.
    session.data = six.b("")
    return response_type