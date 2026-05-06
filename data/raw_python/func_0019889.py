def connect(self, host, port):
        """Connects via a RS-485 to Ethernet adapter."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        self._reader = sock.makefile(mode='rb')
        self._writer = sock.makefile(mode='wb')