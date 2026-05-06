def get_permissions(self):
        """
        :returns: list of dicts, or an empty list if there are no permissions.
        """
        path = Client.urls['all_permissions']
        conns = self._call(path, 'GET')
        return conns