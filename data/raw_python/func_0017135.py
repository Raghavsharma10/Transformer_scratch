def iter_statuses(self, number=-1, etag=None):
        """Iterate over the deployment statuses for this deployment.

        :param int number: (optional), the number of statuses to return.
            Default: -1, returns all statuses.
        :param str etag: (optional), the ETag header value from the last time
            you iterated over the statuses.
        :returns: generator of :class:`DeploymentStatus`\ es
        """
        i = self._iter(int(number), self.statuses_url, DeploymentStatus,
                       etag=etag)
        i.headers = Deployment.CUSTOM_HEADERS
        return i