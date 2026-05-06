def remove_repo(self, repo, team):
        """Remove ``repo`` from ``team``.

        :param str repo: (required), form: 'user/repo'
        :param str team: (required)
        :returns: bool
        """
        for t in self.iter_teams():
            if team == t.name:
                return t.remove_repo(repo)
        return False