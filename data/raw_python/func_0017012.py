def iter_gists(self, username=None, number=-1, etag=None):
        """If no username is specified, GET /gists, otherwise GET
        /users/:username/gists

        :param str login: (optional), login of the user to check
        :param int number: (optional), number of gists to return. Default: -1
            returns all available gists
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Gist <github3.gists.Gist>`\ s
        """
        if username:
            url = self._build_url('users', username, 'gists')
        else:
            url = self._build_url('gists')
        return self._iter(int(number), url, Gist, etag=etag)