def get_connections(self):
        """
        :returns: list of dicts, or an empty list if there are no connections.
        """
        path = Client.urls['all_connections']
        conns = self._call(path, 'GET')
        return conns