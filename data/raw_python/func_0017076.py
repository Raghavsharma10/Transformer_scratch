def create_fork(self, organization=None):
        """Create a fork of this repository.

        :param str organization: (required), login for organization to create
            the fork under
        :returns: :class:`Repository <Repository>` if successful, else None
        """
        url = self._build_url('forks', base_url=self._api)
        if organization:
            resp = self._post(url, data={'organization': organization})
        else:
            resp = self._post(url)
        json = self._json(resp, 202)

        return Repository(json, self) if json else None