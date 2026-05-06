def user(self, login=None):
        """Returns a User object for the specified login name if
        provided. If no login name is provided, this will return a User
        object for the authenticated user.

        :param str login: (optional)
        :returns: :class:`User <github3.users.User>`
        """
        if login:
            url = self._build_url('users', login)
        else:
            url = self._build_url('user')

        json = self._json(self._get(url), 200)
        return User(json, self._session) if json else None