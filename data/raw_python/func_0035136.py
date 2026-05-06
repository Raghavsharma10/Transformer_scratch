def send(self, data):
        """ Open transport, send data, and yield response chunks.
        """
        sock = socket.socket(*self.sock_args)
        try:
            sock.connect(self.sock_addr)
        except socket.error as exc:
            raise socket.error("Can't connect to %r (%s)" % (self.url.geturl(), exc))

        try:
            # Send request
            sock.send(data)

            # Read response
            while True:
                chunk = sock.recv(self.CHUNK_SIZE)
                if chunk:
                    yield chunk
                else:
                    break
        finally:
            # Clean up
            sock.close()