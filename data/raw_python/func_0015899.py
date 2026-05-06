def get_user_permissions(self, username):
        """
        :returns: list of dicts, or an empty list if there are no permissions.

        :param string username: User to set permissions for.
        """

        path = Client.urls['user_permissions'] % (username,)
        conns = self._call(path, 'GET')
        return conns