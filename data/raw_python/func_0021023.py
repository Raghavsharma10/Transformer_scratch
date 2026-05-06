def connect(username, password, host, heartbeats=(0,0)):
    """STOMP connect command.

    username, password:
        These are the needed auth details to connect to the
        message server.

    After sending this we will receive a CONNECTED
    message which will contain our session id.

    """
    if len(heartbeats) != 2:
        raise ValueError('Invalid heartbeat %r' % heartbeats)
    cx, cy = heartbeats
    return "CONNECT\naccept-version:1.1\nhost:%s\nheart-beat:%i,%i\nlogin:%s\npasscode:%s\n\n\x00\n" % (host, cx, cy, username, password)