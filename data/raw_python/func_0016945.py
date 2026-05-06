def team(self, team_id):
        """Returns Team object with information about team specified by
        ``team_id``.

        :param int team_id: (required), unique id for the team
        :returns: :class:`Team <Team>`
        """
        json = None
        if int(team_id) > 0:
            url = self._build_url('teams', str(team_id))
            json = self._json(self._get(url), 200)
        return Team(json, self._session) if json else None