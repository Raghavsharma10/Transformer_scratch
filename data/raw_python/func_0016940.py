def is_public_member(self, login):
        """Check if the user with login ``login`` is a public member.

        :returns: bool
        """
        url = self._build_url('public_members', login, base_url=self._api)
        return self._boolean(self._get(url), 204, 404)