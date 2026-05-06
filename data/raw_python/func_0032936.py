def _create_listening_stream(self, pull_addr):
        """
        Create a stream listening for Requests. The `self._recv_callback`
        method is asociated with incoming requests.
        """
        sock = self._zmq_context.socket(zmq.PULL)
        sock.connect(pull_addr)
        stream = ZMQStream(sock, io_loop=self.io_loop)

        return stream