def _open_connection(self):
        """ Open a new connection socket to the CPS."""
        if self._scheme == 'unix':
            self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0)
            self._connection.connect(self._path)
        elif self._scheme == 'tcp':
            self._connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.SOL_TCP)
            self._connection.connect((self._host, self._port))
        elif self._scheme == 'http':
            self._connection =  httplib.HTTPConnection(self._host, self._port, strict=False)
        else:
            raise ConnectionError("Connection scheme not recognized!")