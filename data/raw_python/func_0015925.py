def create_user(self, username, password, tags=""):
        """
        Creates a user.

        :param string username: The name to give to the new user
        :param string password: Password for the new user
        :param string tags: Comma-separated list of tags for the user
        :returns: boolean
        """
        path = Client.urls['users_by_name'] % username
        body = json.dumps({'password': password, 'tags': tags})
        return self._call(path, 'PUT', body=body,
                                 headers=Client.json_headers)