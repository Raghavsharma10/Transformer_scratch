def connect(self):
    """ Returns a new socket connection to this server. """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", self._port))
    return sock