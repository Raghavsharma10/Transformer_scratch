def publicize_member(self, login):
        """Make ``login``'s membership in this organization public.

        :returns: bool
        """
        url = self._build_url('public_members', login, base_url=self._api)
        return self._boolean(self._put(url), 204, 404)