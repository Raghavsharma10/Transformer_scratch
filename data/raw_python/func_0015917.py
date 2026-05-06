def get_connection(self, name):
        """
        Get a connection by name. To get the names, use get_connections.

        :param string name: Name of connection to get
        :returns dict conn: A connection attribute dictionary.

        """
        name = quote(name, '')
        path = Client.urls['connections_by_name'] % name
        conn = self._call(path, 'GET')
        return conn