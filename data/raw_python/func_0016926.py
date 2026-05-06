def add_repo(self, repo):
        """Add ``repo`` to this team.

        :param str repo: (required), form: 'user/repo'
        :returns: bool
        """
        url = self._build_url('repos', repo, base_url=self._api)
        return self._boolean(self._put(url), 204, 404)