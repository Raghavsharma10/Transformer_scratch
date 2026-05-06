def delete_connection(self, name):
        """
        Close the named connection. The API returns a 204 on success,
        in which case this method returns True, otherwise the
        error is raised.

        :param string name: The name of the connection to delete.
        :returns bool: True on success.
        """
        name = quote(name, '')
        path = Client.urls['connections_by_name'] % name
        self._call(path, 'DELETE')
        return True