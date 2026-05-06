def add_repo(self, repo, team):
        """Add ``repo`` to ``team``.

        .. note::
            This method is of complexity O(n). This iterates over all teams in
            your organization and only adds the repo when the team name
            matches the team parameter above. If you want constant time, you
            should retrieve the team and call ``add_repo`` on that team
            directly.

        :param str repo: (required), form: 'user/repo'
        :param str team: (required), team name
        """
        for t in self.iter_teams():
            if team == t.name:
                return t.add_repo(repo)
        return False