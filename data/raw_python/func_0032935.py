def start(self):
        """
        Start to listen for incoming requests.
        """
        assert not self._started
        self._listening_stream.on_recv(self._recv_callback)
        self._started = True