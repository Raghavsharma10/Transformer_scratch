def iter_user_teams(self, number=-1, etag=None):
        """Gets the authenticated user's teams across all of organizations.

        List all of the teams across all of the organizations to which the
        authenticated user belongs. This method requires user or repo scope
        when authenticating via OAuth.

        :returns: generator of :class:`Team <github3.orgs.Team>` objects
        """
        url = self._build_url('user', 'teams')
        return self._iter(int(number), url, Team, etag=etag)