def iter_deployments(self, number=-1, etag=None):
        """Iterate over deployments for this repository.

        :param int number: (optional), number of deployments to return.
            Default: -1, returns all available deployments
        :param str etag: (optional), ETag from a previous request for all
            deployments
        :returns: generator of
            :class:`Deployment <github3.repos.deployment.Deployment>`\ s
        """
        url = self._build_url('deployments', base_url=self._api)
        i = self._iter(int(number), url, Deployment, etag=etag)
        i.headers.update(Deployment.CUSTOM_HEADERS)
        return i