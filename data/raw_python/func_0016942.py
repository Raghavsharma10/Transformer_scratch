def iter_teams(self, number=-1, etag=None):
        """Iterate over teams that are part of this organization.

        :param int number: (optional), number of teams to return. Default: -1
            returns all available teams.
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Team <Team>`\ s
        """
        url = self._build_url('teams', base_url=self._api)
        return self._iter(int(number), url, Team, etag=etag)