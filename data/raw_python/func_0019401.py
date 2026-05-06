def connect(self):
        """
        Connect will attempt to connect to the NATS server. The url can
        contain username/password semantics.
        """
        self._build_socket()
        self._connect_socket()
        self._build_file_socket()
        self._send_connect_msg()