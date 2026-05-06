def iter_refs(self, subspace='', number=-1, etag=None):
        """Iterates over references for this repository.

        :param str subspace: (optional), e.g. 'tags', 'stashes', 'notes'
        :param int number: (optional), number of refs to return. Default: -1
            returns all available refs
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Reference <github3.git.Reference>`\ s
        """
        if subspace:
            args = ('git', 'refs', subspace)
        else:
            args = ('git', 'refs')
        url = self._build_url(*args, base_url=self._api)
        return self._iter(int(number), url, Reference, etag=etag)