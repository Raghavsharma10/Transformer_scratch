def set_keep_alive(sock, idle=10, interval=5, fails=5):
    """Sets the keep-alive setting for the peer socket.

    :param sock: Socket to be configured.
    :param idle: Interval in seconds after which for an idle connection a keep-alive probes
      is start being sent.
    :param interval: Interval in seconds between probes.
    :param fails: Maximum number of failed probes.
    """
    import sys

    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if sys.platform in ('linux', 'linux2'):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, fails)
    elif sys.platform == 'darwin':
        sock.setsockopt(socket.IPPROTO_TCP, 0x10, interval)
    else:
        # Do nothing precise for unsupported platforms.
        pass