def has_repo(self, repo):
        """Checks if this team has access to ``repo``

        :param str repo: (required), form: 'user/repo'
        :returns: bool
        """
        url = self._build_url('repos', repo, base_url=self._api)
        return self._boolean(self._get(url), 204, 404)