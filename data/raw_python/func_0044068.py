def handshake(timeout, sam_api, max_version):
    """handshake with sam via a socket.socket instance"""
    sock = controller_connect(sam_api, timeout=timeout)
    response = sam_cmd(sock, greet(max_version))
    if response.ok:
        return sock
    else:
        raise HandshakeError("Failed to handshake with SAM: %s" % repr(response))