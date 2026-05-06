def iter_orgs(self, login=None, number=-1, etag=None):
        """Iterate over public organizations for login if provided; otherwise
        iterate over public and private organizations for the authenticated
        user.

        :param str login: (optional), user whose orgs you wish to list
        :param int number: (optional), number of organizations to return.
            Default: -1 returns all available organizations
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of
            :class:`Organization <github3.orgs.Organization>`\ s
        """
        if login:
            url = self._build_url('users', login, 'orgs')
        else:
            url = self._build_url('user', 'orgs')

        return self._iter(int(number), url, Organization, etag=etag)