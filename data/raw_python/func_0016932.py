def revoke_membership(self, username):
        """Revoke this user's team membership.

        :param str username: (required), name of the team member
        :returns: bool
        """
        url = self._build_url('memberships', username, base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)