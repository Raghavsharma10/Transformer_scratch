def _next_port(self, port):
        """
        Return another port from the 5000-5999 range
        """
        port = 5000 + (port + 1) % 1000
        if port == self.port:
            raise DatacatsError('Too many instances running')
        return port