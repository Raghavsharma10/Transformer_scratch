def star(self, login, repo):
        """Star to login/repo

        :param str login: (required), owner of the repo
        :param str repo: (required), name of the repo
        :return: bool
        """
        resp = False
        if login and repo:
            url = self._build_url('user', 'starred', login, repo)
            resp = self._boolean(self._put(url), 204, 404)
        return resp