def delete_user(self, username):
        """
        Deletes a user from the server.

        :param string username: Name of the user to delete from the server.
        """
        path = Client.urls['users_by_name'] % username
        return self._call(path, 'DELETE')