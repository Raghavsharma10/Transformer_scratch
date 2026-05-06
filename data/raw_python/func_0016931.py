def remove_member(self, login):
        """Remove ``login`` from this team.

        :param str login: (required), login of the member to remove
        :returns: bool
        """
        warnings.warn(
            'This is no longer supported by the GitHub API, see '
            'https://developer.github.com/changes/2014-09-23-one-more-week'
            '-before-the-add-team-member-api-breaking-change/',
            DeprecationWarning)
        url = self._build_url('members', login, base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)