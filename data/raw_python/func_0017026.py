def organization(self, login):
        """Returns a Organization object for the login name

        :param str login: (required), login name of the org
        :returns: :class:`Organization <github3.orgs.Organization>`
        """
        url = self._build_url('orgs', login)
        json = self._json(self._get(url), 200)
        return Organization(json, self) if json else None