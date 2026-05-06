def refresh(self, conditional=False):
        """Re-retrieve the information for this object and returns the
        refreshed instance.

        :param bool conditional: If True, then we will search for a stored
            header ('Last-Modified', or 'ETag') on the object and send that
            as described in the `Conditional Requests`_ section of the docs
        :returns: self

        The reasoning for the return value is the following example: ::

            repos = [r.refresh() for r in g.iter_repos('kennethreitz')]

        Without the return value, that would be an array of ``None``'s and you
        would otherwise have to do: ::

            repos = [r for i in g.iter_repos('kennethreitz')]
            [r.refresh() for r in repos]

        Which is really an anti-pattern.

        .. versionchanged:: 0.5

        .. _Conditional Requests:
            http://developer.github.com/v3/#conditional-requests
        """
        headers = {}
        if conditional:
            if self.last_modified:
                headers['If-Modified-Since'] = self.last_modified
            elif self.etag:
                headers['If-None-Match'] = self.etag

        headers = headers or None
        json = self._json(self._get(self._api, headers=headers), 200)
        if json is not None:
            self.__init__(json, self._session)
        return self