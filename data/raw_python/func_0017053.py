def iter_orgs(self, number=-1, etag=None):
        """Iterate over organizations the user is member of

        :param int number: (optional), number of organizations to return.
            Default: -1 returns all available organization
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: list of :class:`Event <github3.orgs.Organization>`\ s
        """
        # Import here, because a toplevel import causes an import loop
        from .orgs import Organization
        url = self._build_url('orgs', base_url=self._api)
        return self._iter(int(number), url, Organization, etag=etag)