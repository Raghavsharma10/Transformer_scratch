def conceal_member(self, login):
        """Conceal ``login``'s membership in this organization.

        :returns: bool
        """
        url = self._build_url('public_members', login, base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)