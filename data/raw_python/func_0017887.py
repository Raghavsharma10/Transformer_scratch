def _get_client(self, server):
        """
        Get the pyrad client for a given server. RADIUS server is described by
        a 3-tuple: (<hostname>, <port>, <secret>).
        """
        return Client(
            server=server[0],
            authport=server[1],
            secret=server[2],
            dict=self._get_dictionary(),
        )