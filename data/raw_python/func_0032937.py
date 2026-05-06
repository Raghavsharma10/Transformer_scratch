def _create_sending_stream(self, pub_addr):
        """
        Create a `ZMQStream` for sending responses back to Mongrel2.
        """
        sock = self._zmq_context.socket(zmq.PUB)
        sock.setsockopt(zmq.IDENTITY, self.sender_id)
        sock.connect(pub_addr)
        stream = ZMQStream(sock, io_loop=self.io_loop)

        return stream