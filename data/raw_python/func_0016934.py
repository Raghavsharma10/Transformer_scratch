def add_member(self, login, team):
        """Add ``login`` to ``team`` and thereby to this organization.

        .. warning::
            This method is no longer valid. To add a member to a team, you
            must now retrieve the team directly, and use the ``invite``
            method.

        Any user that is to be added to an organization, must be added
        to a team as per the GitHub api.

        .. note::
            This method is of complexity O(n). This iterates over all teams in
            your organization and only adds the user when the team name
            matches the team parameter above. If you want constant time, you
            should retrieve the team and call ``add_member`` on that team
            directly.

        :param str login: (required), login name of the user to be added
        :param str team: (required), team name
        :returns: bool
        """
        warnings.warn(
            'This is no longer supported by the GitHub API, see '
            'https://developer.github.com/changes/2014-09-23-one-more-week'
            '-before-the-add-team-member-api-breaking-change/',
            DeprecationWarning)
        for t in self.iter_teams():
            if team == t.name:
                return t.add_member(login)
        return False