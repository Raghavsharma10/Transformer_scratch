def token(self, token):
        """Set the oauth token and the current_user

        :param token: the oauth token
        :type token: :class:`dict`
        :returns: None
        :rtype: None
        :raises: None
        """
        self._token = token
        if token:
            self.current_user = self.query_login_user()