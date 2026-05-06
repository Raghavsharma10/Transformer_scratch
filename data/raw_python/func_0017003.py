def is_subscribed(self, login, repo):
        """Check if the authenticated user is subscribed to login/repo.

        :param str login: (required), owner of repository
        :param str repo: (required), name of repository
        :returns: bool
        """
        json = False
        if login and repo:
            url = self._build_url('user', 'subscriptions', login, repo)
            json = self._boolean(self._get(url), 204, 404)
        return json