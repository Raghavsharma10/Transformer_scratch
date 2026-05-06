def get_socket(self, inbound):
        """Retrieve a socket used by this connection

        When inbound is True, then the socket from which this connection reads
        data is retrieved. Otherwise the socket to which this connection writes
        data is retrieved.

        Read and write sockets differ depending on whether this is a server- or
        a client-side connection, and on whether a routing demux is in use.
        """

        if inbound and hasattr(self, "_rsock"):
            return self._rsock
        return self._sock