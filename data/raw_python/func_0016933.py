def remove_repo(self, repo):
        """Remove ``repo`` from this team.

        :param str repo: (required), form: 'user/repo'
        :returns: bool
        """
        url = self._build_url('repos', repo, base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)