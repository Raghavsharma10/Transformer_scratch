def create_pgroup_snapshot(self, source, **kwargs):
        """Create snapshot of pgroup from specified source.

        :param source: Name of pgroup of which to take snapshot.
        :type source: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST pgroup**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the created snapshot.
        :rtype: ResponseDict

        .. note::

            Requires use of REST API 1.2 or later.

        """
        # In REST 1.4, support was added for snapshotting multiple pgroups. As a
        # result, the endpoint response changed from an object to an array of
        # objects. To keep the  response type consistent between REST versions,
        # we unbox the response when creating a single snapshot.
        result = self.create_pgroup_snapshots([source], **kwargs)
        if self._rest_version >= LooseVersion("1.4"):
            headers = result.headers
            result = ResponseDict(result[0])
            result.headers = headers
        return result