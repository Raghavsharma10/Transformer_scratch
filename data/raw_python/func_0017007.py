def iter_authorizations(self, number=-1, etag=None):
        """Iterate over authorizations for the authenticated user. This will
        return a 404 if you are using a token for authentication.

        :param int number: (optional), number of authorizations to return.
            Default: -1 returns all available authorizations
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Authorization <Authorization>`\ s
        """
        url = self._build_url('authorizations')
        return self._iter(int(number), url, Authorization, etag=etag)